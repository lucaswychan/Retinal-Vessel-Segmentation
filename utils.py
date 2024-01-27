import matplotlib.pyplot as plt
import numpy as np


# Helper functions for visualization
def visualize_side_by_side(
    img1,
    img1caption,
    img2,
    img2caption,
    img_name,
    img2range=(0, 255),
    img1range=(0, 255),
):

    fig, axs = plt.subplots(1, 2, figsize=(8, 16))
    axs[0].imshow(img1, cmap="gray", vmax=img1range[1], vmin=img1range[0])
    axs[1].imshow(img2, cmap="gray", vmax=img2range[1], vmin=img2range[0])
    axs[0].set_title(img1caption)
    axs[1].set_title(img2caption)
    fig.savefig("Plot/%s.pdf" % img_name, dpi=1000, bbox_inches="tight")
    # fig.show()


def visualize_side_by_3(
    img1,
    img1caption,
    img2,
    img2caption,
    img3,
    img3caption,
    img_name,
    img1range=(0, 255),
    img2range=(0, 255),
    img3range=(0, 255),
):

    fig, axs = plt.subplots(1, 3, figsize=(16, 24))
    axs[0].imshow(img1, cmap="gray", vmax=img1range[1], vmin=img1range[0])
    axs[1].imshow(img2, cmap="gray", vmax=img2range[1], vmin=img2range[0])
    axs[2].imshow(img3, cmap="gray", vmax=img3range[1], vmin=img3range[0])
    axs[0].set_title(img1caption)
    axs[1].set_title(img2caption)
    axs[2].set_title(img3caption)
    fig.savefig("Plot/%s.pdf" % img_name, dpi=1000, bbox_inches="tight")
    # fig.show()


def threshold(val_preds, thresh_value):
    """Threshold the given predicted mask array.

    Parameters
    ----------
    val_preds : np.ndarray
        Predicted segmentation array on validation data
    thresh_value : float

    Returns
    ----------
    np.ndarray
        Thresholded val_preds
    """

    val_preds_thresh = val_preds >= thresh_value

    return val_preds_thresh.astype(int)


def dice_coef(mask1, mask2):
    """Calculate the dice coeffecient score between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        binary mask that consists of either 0 or 1.
    mask2 : np.ndarray
        binary mask that consists of either 0 or 1.

    Returns
    ----------
    float
        dice coefficient between mask1 and mask2.
    """

    dice_coef_score = (2 * np.sum(mask1 * mask2)) / (np.sum(mask1) + np.sum(mask2))

    return dice_coef_score


def avg_dice(y_val, val_preds_thresh):
    """Calculates the average dice coefficient score across all thresholded predictions & label pair of the validation dataset.

    Parameters
    ----------
    y_val : np.ndarray
        Ground truth segmentation labels array of the validation dataset
    val_preds : np.ndarray
        Predicted segmentation masks array on the validation dataset

    Returns
    ----------
    float
        Average dice score coefficient.
    """

    # dicecoef = [dice_coef(y_val[i], val_preds_thresh[i]) for i in range(len(y_val))]

    dicecoef = [
        dice_coef(label, predict) for label, predict in zip(y_val, val_preds_thresh)
    ]
    # dicecoef = np.array(dicecoef)
    average_dice = np.mean(dicecoef)

    """
    average_dice = 0
    n = y_val.shape[0]
    for i in range(n):
      average_dice += dice_coef(y_val[i], val_preds_thresh[i])
    average_dice /= n
    """

    return average_dice


if __name__ == "__main__":
    retinal_vessel_data = np.load("retinal_vessel_dataset.npz")
    implementation_check = np.load("sample_data.npz")
    x_train_raw = retinal_vessel_data["x_train"][..., np.newaxis]
    y_train = retinal_vessel_data["y_train"][..., np.newaxis].astype(int)
    x_val_raw = retinal_vessel_data["x_val"][..., np.newaxis]
    y_val = retinal_vessel_data["y_val"][..., np.newaxis].astype(int)
    print("x_train_raw.shape", x_train_raw.shape)  # Each patch has a shape 64 X 64.
    print("y_train.shape", y_train.shape)
    print("x_val_raw.shape", x_val_raw.shape)
    print("y_val.shape", y_val.shape)
    visualize_side_by_side(
        x_train_raw[21, ...], "image", y_train[21, ...], "label", img2range=(0, 1)
    )
    visualize_side_by_side(
        x_train_raw[121, ...], "image", y_train[121, ...], "label", img2range=(0, 1)
    )
