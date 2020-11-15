#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> top_pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get height
    int64_t height = input.size(2);

    // Copy the last column
    at::Tensor input_temp  = input.select(2, height - 1);
    at::Tensor output_temp = output.select(2, height - 1);
    output_temp.copy_(input_temp);

    at::Tensor max_temp;
    for (int64_t ind = 1; ind < height; ++ind) {
        input_temp  = input.select(2, height - ind - 1);
        output_temp = output.select(2, height - ind);
        max_temp    = output.select(2, height - ind - 1);

        at::max_out(max_temp, input_temp, output_temp);
    }

    return { 
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &top_pool_forward, "Top Pool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
}