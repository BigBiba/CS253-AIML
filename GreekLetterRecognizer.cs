using NeuralNetwork1;
using System;
using System.Drawing;

public class GreekLetterRecognizer
{
    private GreekLetterProcessor processor;
    private BaseNetwork network;
    private bool isTrained = false;

    public GreekLetterRecognizer()
    {
        processor = new GreekLetterProcessor();
        int[] structure = { 1600, 800, 400, 11 };
        network = new StudentNetwork(structure);
    }

    public void Train(int epochs)
    {
        var trainSet = processor.GetTrainDataset(1100);
        network.TrainOnDataSet(trainSet, epochs, 0.01, true);
        isTrained = true;
    }

    public AlphabetLetter Recognize(Bitmap image)
    {
        if (!isTrained)
            throw new InvalidOperationException("Модель не обучена!");

        var sample = processor.GetSample(image);
        return network.Predict(sample);
    }

    public double TestAccuracy(int testSamplesCount = 200)
    {
        var testSet = processor.GetTrainDataset(testSamplesCount);
        double accuracy = testSet.TestNeuralNetwork(network);

        return accuracy;
    }

    public void SaveModel(string filePath)
    {
        network.Save(filePath);
    }

    public void LoadModel(string filePath)
    {
        network.Load(filePath);
        isTrained = true;
    }
}