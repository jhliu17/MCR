import pickle
import torch

from collections import defaultdict


def reduce_grads(params):
    grads = [p.grad.data for p in params if p.requires_grad]
    flat_grads = grads[0].new(sum(g.numel() for g in grads)).zero_()

    # Flatten all gradients
    offset = 0
    for g in grads:
        flat_grads[offset: offset + g.numel()].copy_(g.view(-1))
        offset += g.numel()

    # Reduce across all processes
    torch.distributed.all_reduce(flat_grads)

    # Update gradients
    offset = 0
    for g in grads:
        g.copy_(flat_grads[offset: offset + g.numel()].view_as(g))
        offset += g.numel()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [torch.cuda.ByteTensor(max_size) for i in range(world_size)]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2: enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    result = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()
        result.append(pickle.loads(bytes(out_buffer[2: size + 2].tolist())))
    return result


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    # Assign a unique ID to each module instance, so that incremental state is not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def make_positions(tensor, pad_idx):
    """Replace non-padding symbols with their position numbers. Position numbers begin at pad_idx + 1."""
    max_pos = pad_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(pad_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(pad_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def generate_seq_mask(seq_lens, max_length: int = None) -> torch.Tensor:
    seq_max_len = seq_lens.max() if max_length is None else max_length
    mask = torch.arange(0, seq_max_len).view(1, -1).to(seq_lens.device)
    seq_lens = seq_lens.view(-1, 1)
    mask = mask < seq_lens  # (batch, max_len)
    return mask


def flatten_all(tensor_list: list):
    bz = tensor_list[0].size(0)
    return [i.contiguous().view(bz, -1) for i in tensor_list]


def contains_nan(tensors):
    if isinstance(tensors, list):
        return any([i.isnan().any() for i in tensors])
    else:
        return tensors.isnan().any()
