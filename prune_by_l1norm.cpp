#include <caffe/caffe.hpp>

#define USE_PRUNE

const float kKernelDataThreshold = 0.001;
#ifdef USE_PRUNE
const int VALUE = 10000;
#else
const int VALUE = 0;
#endif

struct KernelInfo {
  int kernel_num;
  int kernel_channels, kernel_w, kernel_h;
  int kernel_dim, kernel_size;

  KernelInfo(int n, int c, int w, int h) {
    kernel_num = n;
    kernel_channels = c;
    kernel_w = w;
    kernel_h = h;
    kernel_dim = c * w * h;
    kernel_size = w * h;
  }
  KernelInfo() {};
};

void remove_file(const char *file_path) {
  if (access(file_path, 0) != -1) {
    remove(file_path);
  }
}

void pruneConvLayer() {

}

void pruneDwConvLayer() {

}

void pruneInnerLayer() {

}

int main() {
  google::InstallFailureSignalHandler();
//    std::string deploy_path = "/home/zt/Documents/examples/mbv1_ssd_voc/MobileNetSSD_deploy.prototxt";
//    std::string model_path = "/home/zt/Documents/examples/mbv1_ssd_voc/MobileNetSSD_deploy.caffemodel";
  std::string deploy_path = "/home/scw/Downloads/ZTE/model/TestModel.prototxt";
  std::string model_path = "/home/scw/Downloads/ZTE/model/TestModel.caffemodel";
//    std::string deploy_path = "/home/zt/Documents/examples/mbv2/2.prototxt";
//    std::string model_path = "/home/zt/Documents/examples/mbv2/2.caffemodel";
//    std::string deploy_path = "/home/zt/Documents/examples/pva_cl/pva_cl_comp.pbtxt";
//    std::string model_path = "/home/zt/Documents/examples/pva_cl/pva_cl_comp.caffemodel";

  caffe::NetParameter input_deploy, input_model;
  caffe::ReadProtoFromBinaryFileOrDie(model_path, &input_model);
  caffe::ReadProtoFromTextFileOrDie(deploy_path, &input_deploy);

  int last_output_num = 3;
  std::set<int> last_layer_pruned_kernel_index, cur_layer_pruned_kernel_index;
  KernelInfo last_kernel_info, kernel_info;
  int last_conv_id = -1;

  // be careful of the layer_size and layers_size
  for (int i = 0; i < input_model.layer_size(); ++i) {
    caffe::LayerParameter *source_layer = input_model.mutable_layer(i);
    LOG(INFO) << source_layer->name() << "  " << source_layer->type();

    // cut the conv layers
    if (source_layer->type() == "Convolution" || source_layer->type() == "DepthwiseConvolution") {
      bool is_dw_conv = false;
      if (source_layer->convolution_param().group() != 1) {
        is_dw_conv = true;
      }

      last_kernel_info = kernel_info;
      kernel_info = KernelInfo(source_layer->blobs(0).shape().dim(0), source_layer->blobs(0).shape().dim(1),
                               source_layer->blobs(0).shape().dim(2), source_layer->blobs(0).shape().dim(3));
      last_layer_pruned_kernel_index = cur_layer_pruned_kernel_index;
      cur_layer_pruned_kernel_index.clear();

      if (source_layer->blobs_size() == 1) {
        // wait, todo
      } else if (source_layer->blobs_size() == 2) { // 2 blobs means having bias
        for (int k = 0; k < kernel_info.kernel_num; ++k) {
          int blob_data_index = k * kernel_info.kernel_dim;

          // 丢卷积核
          float kernel_data_sum = 0.0;
          for (int j = 0; j < kernel_info.kernel_dim; ++j) {
            kernel_data_sum += fabs(source_layer->blobs(0).data(blob_data_index + j));
          }
//                    LOG(INFO) << kernel_info.kernel_dim << "  " << kernel_data_sum;
          if (kernel_data_sum < kKernelDataThreshold * kernel_info.kernel_dim) {
            cur_layer_pruned_kernel_index.insert(k);
            // change this kernel's value
            for (int j = 0; j < kernel_info.kernel_dim; ++j) {
              source_layer->mutable_blobs(0)->set_data(blob_data_index + j, VALUE);
            }
            source_layer->mutable_blobs(1)->set_data(k, VALUE);
          }
          // 对于普通conv，丢上一级丢弃的核对应的通道
          if (!is_dw_conv) {
            for (auto pruning_channel_index : last_layer_pruned_kernel_index) {
              for (int j = 0; j < kernel_info.kernel_size; ++j) {
                source_layer->mutable_blobs(0)->set_data(
                    blob_data_index + pruning_channel_index * kernel_info.kernel_size + j, VALUE);
              }
            }
          } else { // 对于dw层，丢弃的是对应通道的卷积核
            if (last_layer_pruned_kernel_index.count(k) == 1) {
              cur_layer_pruned_kernel_index.insert(k);
              for (int j = 0; j < kernel_info.kernel_dim; ++j) {
                source_layer->mutable_blobs(0)->set_data(blob_data_index + j, VALUE);
              }
              source_layer->mutable_blobs(1)->set_data(k, VALUE);
            }
          }
        }

        if (is_dw_conv && !cur_layer_pruned_kernel_index.empty()) { // 如果是dw层则要相应丢弃上一层的对应卷积核
          std::set<int> tmp_set;
          tmp_set.insert(last_layer_pruned_kernel_index.begin(), last_layer_pruned_kernel_index.end());
          tmp_set.insert(cur_layer_pruned_kernel_index.begin(), cur_layer_pruned_kernel_index.end());
          //如果是上一层已经丢弃的则不需要进行处理
          CHECK(tmp_set.size() >= last_layer_pruned_kernel_index.size()) << "It must be wrong.";
          if (tmp_set.size() > last_layer_pruned_kernel_index.size()) {
            caffe::LayerParameter *tmp_last_layer = input_model.mutable_layer(last_conv_id);
            for (auto tmp_index : cur_layer_pruned_kernel_index) {
              for (int j = 0; j < last_kernel_info.kernel_dim; ++j) {
                tmp_last_layer->mutable_blobs(0)->set_data(
                    tmp_index * last_kernel_info.kernel_dim + j, VALUE);
              }
              tmp_last_layer->mutable_blobs(1)->set_data(tmp_index, VALUE);
            }

#ifdef USE_PRUNE
            int tmp_remain = last_kernel_info.kernel_num - tmp_set.size();
            // update
            tmp_last_layer->mutable_blobs(0)->mutable_shape()->set_dim(0, tmp_remain);
            tmp_last_layer->mutable_blobs(1)->mutable_shape()->set_dim(0, tmp_remain);
            tmp_last_layer->mutable_convolution_param()->set_num_output(tmp_remain);
            for (int m = 0; m < input_deploy.layer_size(); ++m) {
              if (input_deploy.layer(m).name() == tmp_last_layer->name()) {
                input_deploy.mutable_layer(m)->mutable_convolution_param()->set_num_output(tmp_remain);
                if (input_deploy.layer(m).convolution_param().group() != 1) {
                  input_deploy.mutable_layer(m)->mutable_convolution_param()->set_group(tmp_remain);
                }
                break;
              }
            }
#endif

            LOG(INFO) << "Reprun " << tmp_set.size() - last_layer_pruned_kernel_index.size()
                      << " kernels in layer: " << tmp_last_layer->name();
          }
        }

#ifdef USE_PRUNE
        int remain_kernel_num = kernel_info.kernel_num - cur_layer_pruned_kernel_index.size();
        // update the weight file
        source_layer->mutable_blobs(0)->mutable_shape()->set_dim(0, remain_kernel_num);
        source_layer->mutable_blobs(1)->mutable_shape()->set_dim(0, remain_kernel_num);
        source_layer->mutable_convolution_param()->set_num_output(remain_kernel_num);
        if (is_dw_conv) {
          source_layer->mutable_convolution_param()->set_group(remain_kernel_num);
        } else {
          source_layer->mutable_blobs(0)->mutable_shape()->set_dim(1, last_output_num);
        }
        // update the deploy file
        if (!cur_layer_pruned_kernel_index.empty()) {
          for (int m = 0; m < input_deploy.layer_size(); ++m) {
            if (input_deploy.layer(m).name() == source_layer->name()) {
              input_deploy.mutable_layer(m)->mutable_convolution_param()->set_num_output(remain_kernel_num);
              if (input_deploy.layer(m).convolution_param().group() != 1) {
                input_deploy.mutable_layer(m)->mutable_convolution_param()->set_group(remain_kernel_num);
              }
              break;
            }
          }
        }
        last_output_num = remain_kernel_num;
#endif
        LOG(INFO) << "Prune " << cur_layer_pruned_kernel_index.size() << " kernels in layer: " << source_layer->name();
      } else {
        LOG(FATAL) << "Something wrong in your weights file";
      }

      last_conv_id = i;
    }
  }

#ifdef USE_PRUNE
  remove_file((model_path + ".tmp").c_str());
  caffe::WriteProtoToTextFile(input_model, model_path + ".tmp");

  // use ifstream to trans the tmp txt file
  std::ifstream intermediate_results((model_path + ".tmp").c_str());
  std::ofstream out_intermediate_results((model_path + ".tmp2").c_str());
  CHECK(intermediate_results.is_open() && out_intermediate_results.is_open())
  << "The intermediate_results file cannot open.";

  std::string line;
  while (std::getline(intermediate_results, line)) {
    if (line != "    data: 10000") {
      out_intermediate_results << line << std::endl;
    }
  }
  intermediate_results.close();
  out_intermediate_results.close();

  caffe::NetParameter output_model;
  caffe::ReadProtoFromTextFileOrDie(model_path + ".tmp2", &output_model);

  remove_file((model_path + ".new").c_str());
  remove_file((deploy_path + ".new").c_str());
  caffe::WriteProtoToBinaryFile(output_model, model_path + ".new");
  caffe::WriteProtoToTextFile(input_deploy, deploy_path + ".new");

  remove((model_path + ".tmp").c_str());
  remove((model_path + ".tmp2").c_str());
#else
  caffe::WriteProtoToBinaryFile(input_model, model_path + ".new");
  caffe::WriteProtoToTextFile(input_deploy, deploy_path + ".new");
#endif

  return 0;
}
