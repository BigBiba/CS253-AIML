using NeuralNetwork1;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;        
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        
        GreekLetterRecognizer recognizer;        
        private const string ModelPath = "greek_model.bin";

        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService)
        {
            aiml = aimlService;
            client = new TelegramBotClient(token);

            recognizer = new GreekLetterRecognizer();
            
            if (TryLoadModel())
            {
                Console.WriteLine("Модель загружена успешно");
                double accuracy = recognizer.TestAccuracy(200);
                Console.WriteLine($"Точность модели: {accuracy:P2}");
            }
            else
            {
                Console.WriteLine("Модель не найдена, требуется обучение...");                
                TrainModel();
            }

            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);            
            var res = client.GetMeAsync().Result;
            Username = res.Username;
        }

        private bool TryLoadModel()
        {
            try
            {
                if (System.IO.File.Exists(ModelPath))
                {
                    recognizer.LoadModel(ModelPath);                    
                    return true;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка загрузки модели: {ex.Message}");
            }
            return false;
        }

        private void TrainModel()
        {
            Console.WriteLine("Начинаем обучение модели...");
            recognizer.Train(120);
            Console.WriteLine("Модель обучена");
            double accuracy = recognizer.TestAccuracy(200);
            Console.WriteLine($"Точность после обучения: {accuracy:P2}");
            
            recognizer.SaveModel(ModelPath);            
            Console.WriteLine("Модель сохранена");
        }        
        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;                
                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");
                var aimlResponse = aiml.Talk(chatId, username, messageText) ?? string.Empty;
                aimlResponse = Regex.Replace(aimlResponse, "\\s+", " ").Trim();

                if (aimlResponse.Length == 0)
                    aimlResponse = "Я тебя не понимаю";

                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aimlResponse,
                    cancellationToken: cancellationToken);
                return;
            }            
            if (message.Type == MessageType.Photo)
            {
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = client.GetFileAsync(photoId).Result;
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                var img = System.Drawing.Image.FromStream(imageStream);

                Bitmap originalBm = new Bitmap(img);
                
                double[] input = Helpers.ToInput(originalBm);

                using (Bitmap processed = originalBm.ToInputBitmap())
                {
                    using (MemoryStream ms = new MemoryStream())
                    {
                        processed.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                        ms.Position = 0;

                        AlphabetLetter recognized = recognizer.Recognize(originalBm);

                        await client.SendPhotoAsync(
                            message.Chat.Id,
                            ms,
                            $"Распознана буква: {recognized}\n(Выделенный и обработанный блоб)",
                            cancellationToken: cancellationToken
                        );
                    }
                }                                
            }        
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {                        
            cts.Cancel();
        }
    }
}
