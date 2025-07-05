import argparse
from pathlib import Path

import torch
from torchvision.io import decode_image

import _common



# TODO: test this
def _round_preserve_sum(t: torch.Tensor, target_sum: float) -> torch.Tensor:
    """
    Tries fairly rounding tensor values whilst preserving their sum
    """
    if torch.any(t < 0):
        raise NotImplementedError(f"{t=} contains negative values which are not supported")
    abs_t = torch.abs(t)
    init_round = torch.round(abs_t)
    diff = target_sum - init_round.sum()
    fracs = abs_t - torch.floor(abs_t)
    ranks = torch.where(fracs < 0.5, fracs, 1 - fracs)
    n_adjust = int(abs(diff))
    if diff > 0:
        _, indices = torch.topk(ranks, n_adjust)
        init_round[indices] += 1
    elif diff < 0:
        _, indices = torch.topk(ranks, n_adjust)
        init_round[indices] -= 1

    return init_round

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=argparse.FileType("rb"),
        default="mnist_cnn.pt",
        help="Path to the saved model"
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default="data_for_inference/",
        help="Path for the image file(s) to run the model on"
    )
    args = parser.parse_args()

    model_state_dict = torch.load(args.model_path)

    exts = ["jpg", "jpeg", "png"]
    exts_pattern = "[" + ",".join(exts) + "]"
    if args.target_path.is_dir():
        img_paths = list(args.target_path.glob(f"**/*.{exts_pattern}*"))
        if len(img_paths) == 0:
            parser.error(f"--target-path={args.target_path} results in no files ending with {exts_pattern}")
    elif args.target_path.is_file():
        for ext in exts:
            if args.target_path.suffix[1:] == ext:
                break
        else:
            parser.error(f"--target-path={args.target_path} does not end with a valid extension {exts_pattern}")
        img_paths = [args.target_path]
    else:
        assert not args.target_path.exists()  # sanity check
        parser.error(f"--target-path={args.target_path} does not exist")

    # TODO: support jpegs
    contains_non_png = False
    filtered_img_paths = []
    for img_path in img_paths:
        if img_path.suffix != ".png":
            contains_non_png = True
        else:
            filtered_img_paths.append(img_path)
    if contains_non_png:
        if len(filtered_img_paths) == 0:
            parser.error(f"--target-path={args.target_path} contains non-png files which aren't supported yet")
        else:
            print(f"WARN: --target-path={args.target_path} contains non-png files which aren't supported yet and so will not be processed")

    net = _common.Net()
    net.load_state_dict(model_state_dict)
    net.eval()

    decoded_images = []
    for p in filtered_img_paths:
        img = decode_image(p)
        normalised_img = img.float() / 255.0
        transformed_img = _common.transform(normalised_img)
        decoded_images.append(transformed_img)

    if len(decoded_images) == 1:
        batch = decoded_images[0].unsqueeze(0)
    else:
        assert len(decoded_images) > 1  # sanity check
        batch = torch.stack(decoded_images)

    with torch.no_grad():
        outputs = net(batch)

    image_digit_probabilities = torch.exp(outputs)
    assert image_digit_probabilities.shape[1] == 10  # sanity check
    digit_probability_sums = image_digit_probabilities.sum(dim=0)
    counts = _round_preserve_sum(digit_probability_sums, batch.shape[0]).int()
    assert counts.shape == (10,)  # sanity check

    print("digit,count")
    for digit, count in zip(range(10), counts.tolist()):
        if count != 0:
            print(f"{digit},{count}")

if __name__ == "__main__":
    main()
