import torch

import matplotlib.pyplot as plt

# Hiển thị ảnh gốc, nhãn thực tế và dự đoán
n = 50  # Số lượng ảnh muốn hiển thị
test_iter = iter(test_dl)
for _ in range(n):
    test_img, test_mask = next(test_iter)
    test_gen_mask = model(test_img)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(test_img.squeeze().cpu().numpy().transpose(1, 2, 0))
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(test_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('True Mask')

    plt.subplot(1, 3, 3)
    binary_test_gen_mask = (torch.sigmoid(test_gen_mask) > 0.5).float()
    plt.imshow(binary_test_gen_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Predicted Mask')

    plt.show()
