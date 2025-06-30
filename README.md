# SAM-Med3d

## SAM
SAM的输入可以是点，框，粗略的画图，也可以是文字：
![image](https://github.com/user-attachments/assets/f63e28fe-e740-42e7-ba27-ab5222957f30)

图像编码器将图像编码为标准形式的embedding，可以选择很多网络，SAM选择了VIT的encoder，可以处理更高分辨率。

对于提示编码器，如果提示输入是密集的，比如一个物体的简单编码，就使用卷积操作，如果是稀疏的，比如点或边界框，就使用位置编码，如果是文本提示，就使用CLIP。

image embedding和prompt embedding会通过元素集的求和操作合并，通过解码器，将嵌入升维到图像本身的大小，从而得到与输入大小匹配的分割掩码。使用的是修改过的transformer解码器。

训练使用了focal loss和dice loss的线性组合，但输出不是只输出一个单一掩码，而是输出多个掩码，这样做是为了消除模糊性，比如说一个点，其实无法说明他到底指的是哪个范围，所以应该训练多个详细程度或者粒度级别不同的掩码。
