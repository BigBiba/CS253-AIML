using NeuralNetwork1;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

public class GreekLetterProcessor
{
    private const string databaseLocation = "";
    private Random random;
    public int LetterCount { get; set; } = 11;

    private Dictionary<AlphabetLetter, List<string>> structure;

    public GreekLetterProcessor()
    {
        random = new Random();
        structure = new Dictionary<AlphabetLetter, List<string>>();
        
        var folderNames = new Dictionary<string, AlphabetLetter>
        {
            { "Alpha", AlphabetLetter.Alpha },
            { "Beta", AlphabetLetter.Beta },
            { "Chi", AlphabetLetter.Chi },
            { "Delta", AlphabetLetter.Delta },
            { "Epsilon", AlphabetLetter.Epsilon },
            { "Eta", AlphabetLetter.Eta },
            { "Gamma", AlphabetLetter.Gamma },
            { "Iota", AlphabetLetter.Iota },
            { "Kappa", AlphabetLetter.Kappa },
            { "Lambda", AlphabetLetter.Lambda },
            { "Mu", AlphabetLetter.Mu }
        };

        foreach (var kvp in folderNames)
        {
            string folderName = kvp.Key;
            AlphabetLetter letter = kvp.Value;

            string letterPath = Path.Combine(databaseLocation, folderName);
            if (Directory.Exists(letterPath))
            {
                structure[letter] = new List<string>();
                
                var pngFiles = Directory.GetFiles(letterPath, "*.png");
                structure[letter].AddRange(pngFiles);

                Console.WriteLine($"Загружено {pngFiles.Length} файлов для буквы {letter}");
            }
        }
    }

    public SamplesSet GetTrainDataset(int count)
    {
        SamplesSet set = new SamplesSet();
        int perClass = count / LetterCount;

        foreach (var letter in structure.Keys)
        {
            var files = structure[letter];
            if (files.Count == 0) continue;

            for (int i = 0; i < Math.Min(perClass, files.Count); i++)
            {
                var samplePath = files[random.Next(files.Count)];
                try
                {
                    using (var bitmap = new Bitmap(samplePath))
                    {
                        double[] input = ExtractFeatures(bitmap);
                        set.AddSample(new Sample(input, LetterCount, letter));
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка загрузки {samplePath}: {ex.Message}");
                }
            }
        }

        set.Shuffle();
        return set;
    }

    public Sample GetSample(Bitmap bitmap)
    {
        double[] input = ExtractFeatures(bitmap);
        return new Sample(input, LetterCount);
    }

    private double[] ExtractFeatures(Bitmap original)
    {        
        return Helpers.ToInput(original);
    }
}