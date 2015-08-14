package com.emma.lr

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler

import com.emma.util.Tools

class LRClassifier {
  
  def main(args: Array[String]) {
    
    println("Please choose a parameter to tune: ")
    println("-1 >> Default")
    println("0 >> Number of iterations ")
    println("1 >> Regulation parameter")
    
    val tune = readLine(">> Input: ")
    
    val conf = new SparkConf().setAppName("Logistic Regression")
    val sc = new SparkContext(conf)
    
    // Parse and process raw data.
    val parsedRawData = Tools.parseData("price_data.txt", sc)    
    val parsedData = Tools.processData(parsedRawData, sc)
    
    // Scale the features
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedData.map { x => x.features })
    val scaledData = parsedData.map { x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))) }
    
    // Train a Logistic Regression model
    var numIters = Array(100, 500, 1000, 1500) // Spark default: 100
    var regParas = Array(1e-5, 1e-4, 1e-3, 0.1, 1)
    
    val count = tune match {
      case "-1" => 1
      case "0" => numIters.length
      case "1" => regParas.length
    }
    
    // Default index value for each parameter
    var iter = 0
    var para = 2

    var accuracy = new Array[Double](count)
    
    // 10-fold cross validation
    val k = 10
    for (i <- 0 to k-1) {
      
      // Split data into training (90%) and test (10%).
      val splits = scaledData.randomSplit(Array(0.9, 0.1), seed = 11L)
      val training = splits(0).cache()
      val test = splits(1)
      
      for (j <- 0 to count-1) {
        
        tune match {
          case "-1" => iter = j
          case "0" => iter = j
          case "1" => para = j
        } 
        
        val lg_model = new LogisticRegressionWithLBFGS()
        lg_model
          .setNumClasses(2)
          .setIntercept(true)
          .optimizer
            .setNumIterations(numIters(iter))
            .setRegParam(regParas(para))
        
        val model = lg_model.run(training)
        
        println(s">>>> Model intercept: ${model.intercept}, weights: ${model.weights}")
        
        // Compute raw scores on the test set.
        val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
        }
        
        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        accuracy(j) += metrics.precision
      }
    }
    
    val acc_avg = accuracy.map(elem => elem/k)
    
    for (i <- 0 to count-1) {
      tune match {
        case "-1" => println("Accuracy for default paras: " + acc_avg(i))
        case "0" => println("Num iteration = " + numIters(i) + " Accuracy = " + acc_avg(i))
        case "1" => println("Max depth = " + regParas(i) + " Accuracy = " + acc_avg(i))
      }
    }
    
    // Save model
    //sc.parallelize(Seq(model), 1).saveAsObjectFile("data/spark-1.3.1/logistic_reg.model")
  }  
  
}