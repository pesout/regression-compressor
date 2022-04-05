# Regression Compressor

This project works with images and combines lossy and lossless compression techniques.

At first, the image is represented as a three-dimensional integer array, from which three two-dimensional arrays representing **each of the color layers are extracted**.

Then, two coefficients and an intercept are obtained using **linear regression**, so that the color values of the blue layer can be computed using only the red and green color layers.

In the red and green layers, **similar values are then represented by their median**. When iterating over the arrays, the **frequencies of occurrence** of each color shade in both layers are calculated.

The 256 color shades, for which the frequencies of occurrence are therefore known, serve as the "alphabet" for the **Huffman tree construction**. This makes it possible to create a dictionary and the original data is translated into binary code, then into an array of bytes, then into characters and **stored in a file**. The necessary information needed for decompression (color frequencies, regression coefficients, etc.) is also stored.

This work also includes a **decompression algorithm** that allows the image to be reconstructed. Clearly, due to the use of lossy compression techniques, the exact original image can no longer be restored.
