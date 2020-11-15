"""
Refer to CenterNet.
Top, bottom, left and right pooling layer just forward max value along with specify direction,
and backward something.
You can use code implemented by cpp to speed up.
"""
import torch


def top_pool_forward(input):
    """input (n, c, h, w)"""
    output = torch.zeros_like(input)

    h = input.size(2)

    # copy last column
    output[:, :, -1] = input[:, :, h - 1]

    for i in range(1, h):
        print(output)
        output[:, :, h - i - 1] = \
            torch.max(input.select(2, h - i - 1), output.select(2, h - i))

    return output


def top_pool_backward(input, grad_out):
    """仅传递最大值的梯度"""
    output = torch.zeros_like(input)

    b, c, h, w = input.size()
    max_val = torch.zeros(b, c, w, dtype=torch.float)
    max_ind = torch.zeros(b, c, w, dtype=torch.long)

    # copy_ method is not deep copy, so data always point to original.
    # you should use clone method to get new data.
    # select method also return shallow copy, so you should remember all method is shallow copy.
    max_val.copy_(input[:, :, h - 1])
    max_ind.fill_(h - 1)

    output_tmp = output[:, :, h - 1]
    grad_out_tmp = grad_out[:, :, h - 1]
    output_tmp.copy_(grad_out_tmp)

    un_max_ind = max_ind.unsqueeze(2)
    gt_mask = torch.zeros(b, c, w, dtype=torch.bool)
    max_tmp = torch.zeros(b, c, w, dtype=torch.float)
    for i in range(1, h):
        input_tmp = input[:, :, h - i - 1]
        # torch.get or at::gt_out in C++ can be replaced by
        #  tensor1 > tensor2.
        # torch.gt(gt_mask, input_tmp, max_val)
        gt_mask = input_tmp > max_val

        # max_val[gt_mask] = input_tmp[gt_mask]
        # max_ind[gt_mask] = h - i - 1

        # torch.masked_select or at::masked_select_out can be replaced
        #  by tensor[mask].
        # torch.scatter_(dim, index, src, reduce=None) -> Tensor
        # 按index填充，dim=0为行填充，dim=1为列填充。
        #  Writes all values from the tensor src into self at the indices specified in the index tensor.
        #  This is the reverse operation of the manner described in gather().
        #  Tensor &at::masked_select_out(Tensor &out, const Tensor &self, const Tensor &mask)
        max_tmp = torch.masked_select(input_tmp, gt_mask)
        # Tensor at::masked_scatter(const Tensor &self, const Tensor &mask, const Tensor &source)
        max_val.masked_scatter_(gt_mask, max_tmp)
        max_ind.masked_fill_(gt_mask, h - i - 1)  # save index of max value

        grad_out_tmp = grad_out[:, :, h - i - 1].unsqueeze(2)
        output.scatter_add_(2, un_max_ind, grad_out_tmp)
        # output[:, :, un_max_ind] += grad_out_tmp

    return output


def bottom_pool_forward(input):
    """input (n, c, h, w)"""
    output = torch.zeros_like(input)

    h = input.size(2)

    # copy last column
    output[:, :, 0] = input[:, :, 0]

    for i in range(h - 1):
        print(output)
        output[:, :, i + 1] = \
            torch.max(input.select(2, i + 1), output.select(2, i))

    return output


def bottom_pool_backward(input, grad_out):
    output = torch.zeros_like(grad_out)
    output[:, :, 0].copy_(grad_out[:, :, 0])

    b, c, h, w = input.size()
    max_val = torch.zeros(b, c, h, dtype=torch.float)
    max_ind = torch.zeros(b, c, h, dtype=torch.long)
    max_val = input[:, :, 0]
    max_ind.fill_(0)

    for i in range(h - 1):
        input_p = input[:, :, i + 1]
        gt_mask = input_p > max_val

        max_val = max_val.masked_scatter(gt_mask, input_p[gt_mask])
        max_ind.masked_fill_(gt_mask, i + 1)

        grad_out_p = grad_out[:, :, i + 1].unsqueeze(2)
        output.scatter_add_(2, max_ind.unsqueeze(2), grad_out_p)

    return output


def right_pool_forward(input):
    pass


def left_pool_forward(input):
    pass


if __name__ == "__main__":
    import top_pool
    import bottom_pool
    f = torch.rand(1, 1, 3, 3)
    g = torch.rand(1, 1, 3, 3)
    print(f, '\n', g)
    print(top_pool_backward(f, g))
    print(bottom_pool_backward(f, g))
    print(top_pool.backward(f, g))
    print(bottom_pool.backward(f, g))
