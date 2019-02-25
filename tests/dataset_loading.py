import datasets

dataset="pugeault"
(x_train, y_train), (x_test, y_test), input_shape,num_classes,labels= datasets.get_data(dataset)

print(f"Images shape {input_shape}")
print(f"classes {num_classes}, labels:\n {labels}")
print(f"Train samples: {y_train.shape[0]}, Test samples: {y_test.shape[0]}")

import matplotlib.pyplot as plt
print(x_train.shape)
initial_sample=0
samples=64
skip= y_train.shape[0] // samples

grid_cols=8
grid_rows=samples // grid_cols
if samples % grid_cols >0:
    grid_rows+=1

f,axes=plt.subplots(grid_rows,grid_cols)
for axr in axes:
    for ax in axr:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

for i in range(samples):
    i_sample=i*skip+initial_sample
    klass = y_train[i_sample].argmax()
    row=i // grid_cols
    col=i % grid_cols
    ax=axes[row,col]
    if input_shape[2]==1:
        ax.imshow(x_train[i_sample,:,:,0], cmap='gray')
    else:
        ax.imshow(x_train[i_sample, :, :,:])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()
