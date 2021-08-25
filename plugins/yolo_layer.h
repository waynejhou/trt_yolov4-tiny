#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include "math_constants.h"
#include "NvInfer.h"

#define MAX_ANCHORS 6

#define CHECK(status)                                           \
    do {                                                        \
        auto ret = status;                                      \
        if (ret != 0) {                                         \
            std::cerr << "Cuda failure in file '" << __FILE__   \
                      << "' line " << __LINE__                  \
                      << ": " << ret << std::endl;              \
            abort();                                            \
        }                                                       \
    } while (0)

namespace Yolo
{
    static constexpr float IGNORE_THRESH = 0.01f;

    struct alignas(float) Detection {
        float bbox[4];  // x, y, w, h
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginV2IOExt
    {
        public:
            YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y);
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin() override = default;

            int getNbOutputs() const noexcept override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

            int initialize() noexcept override;

            void terminate() noexcept override;

            virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0;}

            virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

            virtual size_t getSerializationSize() const noexcept override;

            virtual void serialize(void* buffer) const noexcept override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const noexcept override;

            const char* getPluginVersion() const noexcept override;

            void destroy() noexcept override;

            IPluginV2IOExt* clone() const noexcept override;

            void setPluginNamespace(const char* pluginNamespace) noexcept override;

            const char* getPluginNamespace() const noexcept override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept override TRTNOEXCEPT;

            void detachFromContext() noexcept override;

        private:
            void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

            int mThreadCount = 64;
            int mYoloWidth, mYoloHeight, mNumAnchors;
            float mAnchorsHost[MAX_ANCHORS * 2];
            float *mAnchors;  // allocated on GPU
            int mNumClasses;
            int mInputWidth, mInputHeight;
            float mScaleXY;

            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const noexcept override;

            const char* getPluginVersion() const noexcept override;

            const PluginFieldCollection* getFieldNames() noexcept override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

            void setPluginNamespace(const char* libNamespace) noexcept override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const noexcept override
            {
                return mNamespace.c_str();
            }

        private:
            static PluginFieldCollection mFC;
            static std::vector<nvinfer1::PluginField> mPluginAttributes;
            std::string mNamespace;
    };
};

#endif
