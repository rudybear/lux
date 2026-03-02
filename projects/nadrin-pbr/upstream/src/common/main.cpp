/*
 * Physically Based Rendering
 * Copyright (c) 2017-2018 Michał Siejak
 */

#if !(defined(ENABLE_OPENGL) || defined(ENABLE_VULKAN) || defined(ENABLE_D3D11) || defined(ENABLE_D3D12))
#error "At least one renderer implementation must be enabled via an appropriate ENABLE_* preprocessor macro"
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

#include "application.hpp"

#include "../opengl.hpp"
#include "../vulkan.hpp"
#include "../d3d11.hpp"
#include "../d3d12.hpp"

static void printUsage(const char* argv0)
{
	const std::vector<const char*> flags = {
#if defined(ENABLE_OPENGL)
		"-opengl",
#endif
#if defined(ENABLE_VULKAN)
		"-vulkan",
#endif
#if defined(ENABLE_D3D11)
		"-d3d11",
#endif
#if defined(ENABLE_D3D12)
		"-d3d12",
#endif
	};

	std::fprintf(stderr, "Usage: %s [", argv0);
	for(size_t i=0; i<flags.size(); ++i) {
		std::fprintf(stderr, "%s%s", flags[i], i < (flags.size()-1) ? "|":"");
	}
	std::fprintf(stderr, "]\n");
}

static RendererInterface* createDefaultRenderer()
{
#if defined(ENABLE_D3D11)
	return new D3D11::Renderer;
#elif defined(ENABLE_D3D12)
	return new D3D12::Renderer;
#elif defined(ENABLE_OPENGL)
	return new OpenGL::Renderer;
#elif defined(ENABLE_VULKAN)
	return new Vulkan::Renderer;
#endif
}

static RendererInterface* createNamedRenderer(const std::string& flag)
{
#if defined(ENABLE_OPENGL)
	if(flag == "-opengl") {
		return new OpenGL::Renderer;
	}
#endif
#if defined(ENABLE_VULKAN)
	if(flag == "-vulkan") {
		return new Vulkan::Renderer;
	}
#endif
#if defined(ENABLE_D3D11)
	if(flag == "-d3d11") {
		return new D3D11::Renderer;
	}
#endif
#if defined(ENABLE_D3D12)
	if(flag == "-d3d12") {
		return new D3D12::Renderer;
	}
#endif
	return nullptr;
}

int main(int argc, char* argv[])
{
	RendererInterface* renderer = nullptr;
#if defined(ENABLE_BENCHMARK)
	uint32_t benchmarkFrames = 0;
#endif

	// Parse arguments: renderer flag and optional --benchmark N
	for(int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
#if defined(ENABLE_BENCHMARK)
		if(arg == "--benchmark" && i + 1 < argc) {
			benchmarkFrames = (uint32_t)std::atoi(argv[++i]);
		}
		else
#endif
		if(!renderer) {
			renderer = createNamedRenderer(arg);
			if(!renderer) {
				printUsage(argv[0]);
				return 1;
			}
		}
	}
	if(!renderer) {
		renderer = createDefaultRenderer();
	}

#if defined(ENABLE_BENCHMARK)
	if(benchmarkFrames > 0) {
		renderer->setBenchmarkFrames(benchmarkFrames);
	}
#endif

	try {
		Application().run(std::unique_ptr<RendererInterface>{ renderer });
	}
	catch(const std::exception& e) {
		std::fprintf(stderr, "Error: %s\n", e.what());
		return 1;
	}
}
