package com.emma.rf

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

import com.emma.util.Tools

class RFClassifier {
  def main(args: Array[String]) {
    
    println("Please choose a parameter to tune: ")
    println("-1 >> Default")
    println("0 >> Number of iterations ")
    println("1 >> Max depth")
    println("2 >> Learning rate")
    println("3 >> Max number of bins")
    
    val tune = readLine(">> Input: ")
    
    val conf = new SparkConf().setAppName("Random forest classification")
    val sc = new SparkContext(conf)

    // Parse and process raw data.
    val parsedRawData = Tools.parseData("price_data_f5.txt", sc)    
    val parsedData = Tools.processData(parsedRawData, sc)

    // Train a RandomForest model.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"

    var numTrees = Array(10, 100, 150, 200)
    var maxDepth = Array(6, 8, 10, 12, 14)
    var maxBins = Array(32, 64, 128)
    
    val count = tune match {
      case "-1" => 1
      case "0" => numTrees.length
      case "1" => maxDepth.length
      case _ => maxBins.length
    }
    
    // Default index value for each parameter
    var iter = 2
    var depth = 3
    var bin = 1

    var accuracy = new Array[Double](count)
    
    // 10-fold cross validation
    val k = 10
    for (i <- 0 to k-1) {
      // Split data into training (90%) and test (10%).
      val splits = parsedData.randomSplit(Array((1-1.0/k), 1.0/k), seed = 11L)
      val training = splits(0).cache()
      val test = splits(1)

      for (j <- 0 to count-1) {
        
        tune match {
          case "-1" => iter = j
          case "0" => iter = j
          case "1" => depth = j
          case _   => bin = j
        }   
        
        // index with j for current parameter being tuned
        val model = RandomForest.trainClassifier(training, numClasses, categoricalFeaturesInfo, numTrees(iter), featureSubsetStrategy,
          impurity, maxDepth(depth), maxBins(bin))
      
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
        case "-1" => println("Accuracy for default paras: " + acc_avg(i))
        case "0" => println("Num iteration = " + numTrees(i) + " Accuracy = " + acc_avg(i))
        case "1" => println("Max depth = " + maxDepth(i) + " Accuracy = " + acc_avg(i))
        case _ => println("Max bin = " + maxBins(i) + " Accuracy = " + acc_avg(i))
      }
    }
  }
  
}