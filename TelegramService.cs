using AForge.Imaging.Filters;
using NeuralNetwork1;
using NeuralNetwork1.SampleGeneration;
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
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        private BaseNetwork<AlphabetSampleData> network = new EvgenNetwork<AlphabetSampleData>("..//..//network.net");
        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService)
        {
            aiml = aimlService;
            //var res = client.GetMeAsync().Result;
            //Username = res.Username;
            //aiml.Talk(res.Id, Username, "init");
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            var res = client.GetMeAsync().Result;
            Username = res.Username;
        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;
                //messageText = Regex.Replace(messageText, "ё", "е");
                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");
                var aimlResponse = aiml.Talk(chatId, username, messageText);
                aimlResponse= Regex.Replace(aimlResponse, "\\s+", " ");
                Regex regex = new Regex("^\\s*$");
                if(!regex.IsMatch(aimlResponse))
                // Echo received message text
                    await botClient.SendTextMessageAsync(
                        chatId: chatId,
                        text: aimlResponse,
                        cancellationToken: cancellationToken);
                    return;
            }
            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                var bitmap = new Bitmap(Image.FromStream(imageStream));

                //bitmap = scaleFilter.Apply(bitmap);
                var sample = new Sample<AlphabetSampleData>(bitmap.ToInput(),10, new AlphabetSampleData());


                // Если бы мы хотели получить битмап, то могли бы использовать new Bitmap(Image.FromStream(imageStream))
                // Но вместо этого пошлём картинку назад
                // Стрим помнит последнее место записи, мы же хотим теперь прочитать с самого начала
                //imageStream.Seek(0, 0);
                using (MemoryStream ms = new MemoryStream())
                {
                    bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                    ms.Position = 0;
                    await client.SendPhotoAsync(
                       message.Chat.Id,
                       ms,
                       $"Распознана буква: {network.Predict(sample)}",
                       cancellationToken: cancellationToken
                    );
                }  
                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
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
            // Заканчиваем работу - корректно отменяем задачи в других потоках
            // Отменяем токен - завершатся все асинхронные таски
            cts.Cancel();
        }
    }
}
