using AForge.Imaging;
using AForge.Imaging.Filters;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    static class Helpers
    {
        public static void MutliplyAndApplySigmoid(this double[] vector, double [,] matrix, double[] result)
        {
            var rowsCount = matrix.GetLength(0);
            var colCount = matrix.GetLength(1);
            for (int i = 0; i <colCount ; i++)
            {
                double sum = 0;
                for(int j = 0; j <rowsCount; j++)
                {
                    sum += vector[j] * matrix[j, i];
                }
                result[i] = sum.Sigmoid();
            }  
        }

        public static void MutliplyAndApplySigmoidParallel(this double[] vector, double[,] matrix, double[] result)
        {
            var rowsCount = matrix.GetLength(0);
            var colCount = matrix.GetLength(1);
            for (int i = 0; i < colCount; i++)
            {
                double sum = 0;
                Parallel.For(0, rowsCount, (j) => sum += vector[j] * matrix[j, i]);
                result[i] = sum.Sigmoid();
            }
        }

        public static double Sigmoid(this double value)
        {
            return 1.0 / (Math.Exp(-value) + 1);
        }

        public static Bitmap FindAndExtractMaxBlob(this Bitmap binImage)
        {
            // Настроим алгоритм их поиска и подсчёта
            BlobCounterBase blobCounter = new BlobCounter
            {
                // Включим отсеивание слишком мелких блобов - возможно, нужно подкрутить
                FilterBlobs = true,
                MinWidth = 5,
                MinHeight = 5,
                // Сортировка по размеру, чтобы потом взять наибольший блоб из найденных
                ObjectsOrder = ObjectsOrder.Size
            };
            try
            {
                blobCounter.ProcessImage(binImage);
                Blob[] blobs = blobCounter.GetObjectsInformation();
                // Получаем наибольший блоб: на самом деле зависит от задачи
                // В некоторых случаях нужно несколько блобов (мы распознавали кнопки медиаплеера, нам хватало одной)
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

            // По умолчанию вернём пустую картинку, это костыль чтоб не было null и ошибки нигде не обрабатывать
            // В проде так делать не стоит, но лаба же одноразовая
            return new Bitmap(binImage.Width, binImage.Height);
        }

        public static Bitmap MapBitmap(Bitmap bitmap)
        {
            bitmap = bitmap.FindAndExtractMaxBlob();
            var inputWidth = 20;
            var inputHeight = 20;
            Bitmap tmp = new Bitmap(inputWidth, inputHeight);
            
            using (var g = Graphics.FromImage(tmp))
            {
                    g.Clear(Color.Black);
                    var ratio = bitmap.Width / (double)bitmap.Height;
                    if (ratio <= 1)
                        g.DrawImage(bitmap, new Rectangle(0, 0, (int)(inputWidth * ratio), inputHeight));
                    else
                        g.DrawImage(bitmap, new Rectangle(0, 0, inputWidth, (int)(inputHeight / ratio)));
                    return tmp;
             }
        }


        public static double[] ToInput(this Bitmap bitmap)
        {
            AForge.Imaging.Filters.Grayscale grayFilter = new Grayscale(0.2125, 0.7154, 0.0721);
            var uProcessed = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(bitmap));
            Console.WriteLine("1");
            var threshldFilter = new AForge.Imaging.Filters.OtsuThreshold();

            // threshldFilter.ThresholdValue = 32;
            threshldFilter.ApplyInPlace(uProcessed);
            Console.WriteLine("2");
            Invert InvertFilter = new Invert();
            InvertFilter.ApplyInPlace(uProcessed);
            Console.WriteLine("3");
            bitmap = uProcessed.ToManagedImage();
            bitmap = bitmap.FindAndExtractMaxBlob();
            Console.WriteLine("4");
            var inputWidth = 40;
            var inputHeight = 40;
            using (Bitmap tmp = new Bitmap(inputWidth, inputHeight))
            {
                using (var g = Graphics.FromImage(tmp))
                {
                    g.Clear(Color.Black);
                    var ratio = bitmap.Width / (double)bitmap.Height;
                    if (ratio <= 1)
                        g.DrawImage(bitmap, new Rectangle(0, 0, (int)(inputWidth * ratio), inputHeight));
                    else
                        g.DrawImage(bitmap, new Rectangle(0, 0, inputWidth, (int)(inputHeight / ratio)));
                }
                // заполняем массив, который пойдёт на вход нейросети, масштабируя пиксели из 0..255 в 0..1
                // По идее, у нас ч.б. изображение, и достаточно было бы тернарки вместо деления
                var input = new double[inputHeight * inputWidth];
                for (int i = 0; i < tmp.Width; ++i)
                    for (int j = 0; j < tmp.Height; ++j)
                        input[i * tmp.Height + j] = tmp.GetPixel(i, j).R / 256.0;
                return input;
            }
        }

    }
}
