import argparse
from collections import Counter
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.io import decode_image

import _common


# TODO: test this
def _round_preserve_sum(t: torch.Tensor, target_sum: float) -> torch.Tensor:
    """
    Tries fairly rounding tensor values whilst preserving their sum
    """
    if torch.any(t < 0):
        raise NotImplementedError(f"{t=} contains negative values")
    abs_t = torch.abs(t)
    round_t = torch.round(abs_t)
    diff = target_sum - round_t.sum()
    fracs = abs_t - torch.floor(abs_t)
    ranks = torch.where(fracs < 0.5, fracs, 1 - fracs)
    n_adjust = int(abs(diff))
    if diff > 0:
        _, indices = torch.topk(ranks, n_adjust)
        round_t[indices] += 1
    elif diff < 0:
        _, indices = torch.topk(ranks, n_adjust)
        round_t[indices] -= 1

    return round_t


def main():
    # Define CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=argparse.FileType("rb"),
        default="mnist_cnn.pt",
        help="path to the saved model",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default="data_for_inference/",
        help="path for the image file(s) to run the model on",
    )
    args = parser.parse_args()

    # Load model now to short-circuit issues with --model-path
    model_state_dict = torch.load(args.model)

    # Find images from --target-path
    exts = ["jpg", "jpeg", "png"]
    exts_pattern = "[" + ",".join(exts) + "]"
    if args.target.is_dir():
        img_paths = list(args.target.glob(f"**/*.{exts_pattern}*"))
        if len(img_paths) == 0:
            parser.error(
                f"--target-path={args.target} results in no files ending with "
                f"{exts_pattern}"
            )
    elif args.target.is_file():
        for ext in exts:
            if args.target.suffix[1:] == ext:
                break
        else:
            parser.error(
                f"--target-path={args.target} does not end with a valid "
                f"extension {exts_pattern}"
            )
        img_paths = [args.target]
    else:
        assert not args.target.exists()  # sanity check
        parser.error(f"--target-path={args.target} does not exist")
    decoded_images = [decode_image(p) for p in img_paths]

    # Transform images the same way they were for model training. If necessary,
    # resize images to the same size beforehand (to what appears the most).
    # XXX: maybe they should be resized to how the model was trained?
    dims_counter = Counter(i.shape for i in decoded_images)
    assert all(s[0] == 3 and len(s) == 3 for s in dims_counter.keys())  # sanity check
    mode_shape = dims_counter.most_common(1)[0][0]
    resize_dim = mode_shape[1:]
    resize_transform = transforms.Resize(resize_dim)
    transformed_images = []
    for img in decoded_images:
        if img.shape != mode_shape:
            img = resize_transform(img)
        transformed_img = _common.transform(img)
        transformed_images.append(transformed_img)

    # Setup and run model
    net = _common.Net()
    net.load_state_dict(model_state_dict)
    net.eval()
    if len(decoded_images) == 1:
        batch = transformed_images[0].unsqueeze(0)
    else:
        batch = torch.stack(transformed_images)
    with torch.no_grad():
        outputs = net(batch)

    # Extrapolate model results to counts of digits in images
    image_digit_probabilities = torch.exp(outputs)
    assert image_digit_probabilities.shape[1] == 10  # sanity check
    digit_probability_sums = image_digit_probabilities.sum(dim=0)
    ordered_counts = _round_preserve_sum(digit_probability_sums, batch.shape[0]).int()
    assert ordered_counts.shape == (10,)  # sanity check
    counts_topk = torch.topk(ordered_counts, 10)
    digit_to_desc_count = {}
    for digit, count in zip(counts_topk.indices, counts_topk.values):
        if count != 0:
            digit_to_desc_count[int(digit)] = int(count)

    # Display findings
    print("digit,count")
    for digit, count in digit_to_desc_count.items():
        print(f"{digit},{count}")


if __name__ == "__main__":
    main()
