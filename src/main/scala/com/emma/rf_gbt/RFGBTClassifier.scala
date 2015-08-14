package com.emma.rf_gbt

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.loss.LogLoss
import org.apache.spark.mllib.util.MLUtils

import com.emma.util.Tools

class RFGBTClassifier {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Gradient Boosted Trees")
    val sc = new SparkContext(conf)
    
    // Parse and process raw data.
    val parsedRawData = Tools.parseData("price_data_f5.txt", sc)    
    val parsedData = Tools.processData(parsedRawData, sc)
    
    // Train a RandomForest model.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxBins_rf = 64
    val numTrees = 50
    val maxDepth_rf = 8
    
    // Train a GBT model, initialize with results from RandomForest model.  
    var iterations = Array(10)
    var maxDepth = Array(4, 6, 8, 10, 12)
    var learningRate = Array(1e-4, 1e-3, 0.1, 1)
    var maxBins = Array(32, 64, 128)
    
    val tune = 0 // 0: iteration 1: maxDepth 2: learningRate 3: maxBins
    
    val count = tune match {
      case 0 => iterations.length
      case 1 => maxDepth.length
      case 2 => learningRate.length
      case _ => maxBins.length
    }

    var accuracy = new Array[Double](count)
    
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]() 
    
    // 10-fold validation
    val k = 5
    for (i <- 0 to k-1) {
      
      // Split data into training (90%) and test (10%).
      val splits = parsedData.randomSplit(Array((1-1.0/k), 1.0/k), seed = 11L)
      val training = splits(0).cache()
      val test = splits(1)
      
      val model_rf = RandomForest.trainClassifier(training, numClasses, categoricalFeaturesInfo,
        numTrees, featureSubsetStrategy, impurity, maxDepth_rf, maxBins_rf)
      
      // Get prediction value for all data points from the random forest model
      val initialVal = training.map { point => model_rf.predict(point.features) }
      val features = training.map {case LabeledPoint(label, features) => features}
      
      val training_rf = initialVal.zip(features).map {case (init, feature) => LabeledPoint(init, feature)}
      
      for(j <- 0 to count-1) {
    
        boostingStrategy.numIterations = iterations(j)
        boostingStrategy.treeStrategy.maxDepth = maxDepth(2)
        boostingStrategy.learningRate = learningRate(2)
        boostingStrategy.treeStrategy.maxBins = maxBins(1);
        
        val model = GradientBoostedTrees.train(training_rf, boostingStrategy)
        
       // Evaluate model on test instances and compute test error
        val labelAndPreds = test.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
        accuracy(j) += labelAndPreds.filter(r => r._1 == r._2).count.toDouble / test.count()
      }
    }
    
    val acc_avg = accuracy.map(elem => elem/k)
    
    for (i <- 0 to count-1) {
      tune match {
        case 0 => println("Num iteration = " + iterations(i) + " Accuracy = " + acc_avg(i))
        case 1 => println("Max depth = " + maxDepth(i) + " Accuracy = " + acc_avg(i))
        case 2 => println("Learning rate = " + learningRate(i) + " Accuracy = " + acc_avg(i))
        case _ => println("Max bin = " + maxBins(i) + " Accuracy = " + acc_avg(i))
      }
    }
   
  }
}