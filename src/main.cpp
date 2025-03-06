#define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 0

#include <memory>
#include <stdexcept>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <print>
#include <vulkan/vulkan_raii.hpp>
#include <cmath>

class SDLException final : public std::runtime_error {
public:
    explicit SDLException(const std::string &message) : std::runtime_error(
        std::format("{}: {}", message, SDL_GetError())) {}
};

constexpr auto VULKAN_VERSION{vk::makeApiVersion(0, 1, 4, 0)};

class App {
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window{nullptr, SDL_DestroyWindow};
    bool running{true};

    std::optional<vk::raii::Context> context{};
    std::optional<vk::raii::Instance> instance{};
    std::optional<vk::raii::SurfaceKHR> surface{};
    std::optional<vk::raii::PhysicalDevice> physicalDevice{};
    uint32_t graphicsQueueFamilyIndex{};
    std::optional<vk::raii::Device> device{};
    std::optional<vk::raii::Queue> graphicsQueue{};

    std::optional<vk::raii::CommandPool> commandPool{};
    std::vector<vk::raii::CommandBuffer> commandBuffers{};
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores{};
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores{};
    std::vector<vk::raii::Fence> fences{};

    std::optional<vk::raii::SwapchainKHR> swapchain{};
    std::vector<vk::Image> swapchainImages{};
    vk::Extent2D swapchainExtent{};
    vk::Format swapchainImageFormat{vk::Format::eB8G8R8A8Srgb};
    uint32_t currentSwapchainImageIndex{};

public:
    App() {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw SDLException("Failed to initialize SDL");
        if (!SDL_Vulkan_LoadLibrary(nullptr))
            throw SDLException("Failed to load Vulkan library");
        window.reset(SDL_CreateWindow("Codotaku", 800, 600,
                                      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN));
        if (!window)
            throw SDLException("Failed to create window");
        auto vkGetInstanceProcAddr{reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr())};
        context.emplace(vkGetInstanceProcAddr);
        auto const vulkanVersion{context->enumerateInstanceVersion()};
        std::println("Vulkan {}.{}", VK_API_VERSION_MAJOR(vulkanVersion), VK_API_VERSION_MINOR(vulkanVersion));
    }

    ~App() {
        device->waitIdle();
        SDL_Quit();
    }

    void Init() {
        InitInstance();
        InitSurface();
        PickPhysicalDevice();
        InitDevice();
        InitCommandPool();
        AllocateCommandBuffers();
        InitSyncObjects();
        RecreateSwapchain();
    }

    void Run() {
        SDL_ShowWindow(window.get());
        while (running) {
            HandleEvents();
            Render();
        }
    }

private:
    void Render() {
        vk::Fence const fence{*fences[0]};
        auto _ = device->waitForFences(fence, VK_TRUE, UINT64_MAX);
        device->resetFences(fence);

        auto [acquireResult, imageIndex] = swapchain->
            acquireNextImage(UINT64_MAX, imageAvailableSemaphores[0], nullptr);
        currentSwapchainImageIndex = imageIndex;

        auto const &swapchainImage{swapchainImages[imageIndex]};

        vk::raii::CommandBuffer const &commandBuffer{commandBuffers[0]};
        commandBuffer.reset();
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer.begin(beginInfo);

        auto const t{static_cast<double>(SDL_GetTicks()) * 0.001};

        vk::ClearColorValue const color{
            std::array{static_cast<float>(std::sin(t * 5.0) * 0.5 + 0.5), 0.0f, 0.0f, 1.0f}
        };

        // transfer image layout to transfer destination
        vk::ImageMemoryBarrier const barrier{
            vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            swapchainImage, vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1
            }
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, nullptr, nullptr,
                                      barrier);

        commandBuffer.clearColorImage(swapchainImage, vk::ImageLayout::eTransferDstOptimal, color,
                                      vk::ImageSubresourceRange{
                                          vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1
                                      });

        vk::ImageMemoryBarrier const barrier2{
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            swapchainImage, vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1
            }
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, nullptr, nullptr,
                                      barrier2);

        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBuffers(*commandBuffer);
        submitInfo.setWaitSemaphores(*imageAvailableSemaphores[0]);
        submitInfo.setSignalSemaphores(*renderFinishedSemaphores[0]);
        constexpr vk::PipelineStageFlags waitStage{vk::PipelineStageFlagBits::eTransfer};
        submitInfo.setWaitDstStageMask(waitStage);
        graphicsQueue->submit(submitInfo, fence);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.setSwapchains(**swapchain);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*renderFinishedSemaphores[0]);
        _ = graphicsQueue->presentKHR(presentInfo);
    }

    void HandleEvents() {
        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type) {
                case SDL_EVENT_QUIT:
                    running = false;
                    break;
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    RecreateSwapchain();
                    break;
                default: break;
            }
    }

    void RecreateSwapchain() {
        vk::SurfaceCapabilitiesKHR const surfaceCapabilities{physicalDevice->getSurfaceCapabilitiesKHR(*surface)};
        swapchainExtent = surfaceCapabilities.currentExtent;

        vk::SwapchainCreateInfoKHR swapchainCreateInfo{};
        swapchainCreateInfo.surface = *surface;
        swapchainCreateInfo.minImageCount = surfaceCapabilities.minImageCount + 1;
        swapchainCreateInfo.imageFormat = swapchainImageFormat;
        swapchainCreateInfo.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        swapchainCreateInfo.imageExtent = swapchainExtent;
        swapchainCreateInfo.imageArrayLayers = 1;
        swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eTransferDst;
        swapchainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
        swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        swapchainCreateInfo.presentMode = vk::PresentModeKHR::eMailbox;
        swapchainCreateInfo.clipped = true;

        // todo: figure out why old swapchain handle is invalid
        // if (swapchain.has_value()) swapchainCreateInfo.oldSwapchain = **swapchain;

        swapchain.emplace(*device, swapchainCreateInfo);
        swapchainImages = swapchain->getImages();
    }

    void InitSyncObjects() {
        vk::FenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;

        fences.emplace_back(*device, fenceCreateInfo);

        vk::SemaphoreCreateInfo semaphoreCreateInfo{};
        imageAvailableSemaphores.emplace_back(*device, semaphoreCreateInfo);
        renderFinishedSemaphores.emplace_back(*device, semaphoreCreateInfo);
    }

    void AllocateCommandBuffers() {
        vk::CommandBufferAllocateInfo allocateInfo{};
        allocateInfo.commandPool = *commandPool;
        allocateInfo.commandBufferCount = 1;

        commandBuffers = device->allocateCommandBuffers(allocateInfo);
    }

    void InitCommandPool() {
        vk::CommandPoolCreateInfo commandPoolCreateInfo{};
        commandPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
        commandPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

        commandPool.emplace(*device, commandPoolCreateInfo);
    }

    void InitDevice() {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
        std::array queuePriorities{1.0f};
        queueCreateInfo.setQueuePriorities(queuePriorities);

        vk::DeviceCreateInfo deviceCreateInfo{};
        std::array queueCreateInfos{queueCreateInfo};
        deviceCreateInfo.setQueueCreateInfos(queueCreateInfos);
        std::array<const char* const, 1> enabledExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        deviceCreateInfo.setPEnabledExtensionNames(enabledExtensions);

        device.emplace(*physicalDevice, deviceCreateInfo);

        graphicsQueue.emplace(*device, graphicsQueueFamilyIndex, 0);
    }

    void PickPhysicalDevice() {
        auto const physicalDevices{instance->enumeratePhysicalDevices()};
        if (physicalDevices.empty())
            throw std::runtime_error("No Vulkan devices found");
        physicalDevice.emplace(*instance, *physicalDevices.front());
        auto const rawDeviceName{physicalDevice->getProperties().deviceName};
        std::string deviceName(rawDeviceName.data(), std::strlen(rawDeviceName));
        std::println("{}", deviceName);
        graphicsQueueFamilyIndex = 0;
    }

    void InitInstance() {
        vk::ApplicationInfo applicationInfo{};
        applicationInfo.applicationVersion = VULKAN_VERSION;

        vk::InstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        uint32_t extensionCount;
        instanceCreateInfo.ppEnabledExtensionNames = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        instanceCreateInfo.enabledExtensionCount = extensionCount;

        instance.emplace(*context, instanceCreateInfo);
    }

    void InitSurface() {
        VkSurfaceKHR raw_surface;
        if (!SDL_Vulkan_CreateSurface(window.get(), **instance, nullptr, &raw_surface))
            throw SDLException("Failed to create Vulkan surface");
        surface.emplace(*instance, raw_surface);
    }
};

int main() {
    try {
        App app{};
        app.Init();
        app.Run();
    }
    catch (const SDLException &e) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Error: %s", e.what());
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", e.what(), nullptr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
