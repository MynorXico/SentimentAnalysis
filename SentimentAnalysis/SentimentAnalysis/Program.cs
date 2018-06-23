using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Additional
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;


namespace SentimentAnalysis
{

    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        // Added async to Main with a Task return type because I'm saving the model to a zip
        // file later, and the program needs to wait until that external task completes
        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            Predict(model);
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            // Initialize a new instance of LearningPipeline that will include the data loading, data
            // processing/feturization, and model.
            var pipeline = new LearningPipeline();

            // The TextLoader object is the first part of the pipeline, and loads the training file data
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());

            // TextFeaturizer converts the SentimentText colun into a numeric vector called Features used 
            // bye the machine learning algorithm. This is the prepcoessingfeaturization step. Using additional
            // componentes available in ML.NET can enable better results with the model.
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            pipeline.Add(new FastTreeBinaryClassifier()
            {
                NumLeaves = 5,
                NumTrees = 5,
                MinDocumentsInLeafs = 2
            });

            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();
            await model.WriteAsync(_modelPath);

            return model;
        }

        /// <summary>
        /// Loads the test dataset
        /// Creates a binary evaluator
        /// Evaluates the model and creates metrics
        /// Displays the metrics
        /// </summary>
        /// <param name="model"></param>
        public static void Evaluate(PredictionModel<SentimentData,SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();

            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("-------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
        
        /// <summary>
        /// Creates test data
        /// Predicts sentiment based on test data
        /// Combines test data and predictions for reporting
        /// Dispays the predicted results
        /// </summary>
        /// <param name="model"></param>
        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach(var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive":"Negative")}");
            }
            Console.WriteLine();
        }
    }
}
