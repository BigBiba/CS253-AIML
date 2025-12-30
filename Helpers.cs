using AForge.Imaging;
using AForge.Imaging.Filters;
using System;
using System.Drawing;

namespace NeuralNetwork1
{
    static class Helpers
    {        
        public static Bitmap FindAndExtractMaxBlob(this Bitmap binImage)
        {            
            BlobCounterBase blobCounter = new BlobCounter
            {                
                FilterBlobs = true,
                MinWidth = 5,
                MinHeight = 5,                
                ObjectsOrder = ObjectsOrder.Size
            };
            try
            {
                blobCounter.ProcessImage(binImage);
                Blob[] blobs = blobCounter.GetObjectsInformation();                                
                if (blobs.Length > 0)
                {
                    blobCounter.ExtractBlobsImage(binImage, blobs[0], false);
                    return blobs[0].Image.ToManagedImage();
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            return new Bitmap(binImage.Width, binImage.Height);
        }

        public static Bitmap ToInputBitmap(this Bitmap original)
        {            
            Grayscale grayFilter = new Grayscale(0.2125, 0.7154, 0.0721);
            var uProcessed = grayFilter.Apply(UnmanagedImage.FromManagedImage(original));
            var threshldFilter = new OtsuThreshold();
            threshldFilter.ApplyInPlace(uProcessed);
            Invert InvertFilter = new Invert();
            InvertFilter.ApplyInPlace(uProcessed);
            Bitmap binary = uProcessed.ToManagedImage();

            Bitmap blob = binary.FindAndExtractMaxBlob();
            var dilatation = new Dilatation();
            blob = dilatation.Apply(blob);

            var inputWidth = 40;
            var inputHeight = 40;
            Bitmap resized = new Bitmap(inputWidth, inputHeight);
            using (var g = Graphics.FromImage(resized))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
                var ratio = blob.Width / (double)blob.Height;
                if (ratio <= 1)
                    g.DrawImage(blob, new Rectangle(0, 0, (int)(inputWidth * ratio), inputHeight));
                else
                    g.DrawImage(blob, new Rectangle(0, 0, inputWidth, (int)(inputHeight / ratio)));
            }

            return resized;
        }

        public static double[] ToInput(this Bitmap bitmap)
        {
            Bitmap processed = bitmap.ToInputBitmap();

            double[] input = new double[40 * 40];
            for (int i = 0; i < 40; ++i)
                for (int j = 0; j < 40; ++j)
                    input[i * 40 + j] = processed.GetPixel(i, j).R / 256.0;

            return input;

        }

    }
}
