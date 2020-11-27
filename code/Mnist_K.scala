package main.scala

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
// $example on$
// $example off$
import org.apache.spark.sql.SQLContext


object Mnist_K {

  def compute(center: Array[Double], test: Array[Double]):Double = {

    var error = 0.0
    assert(center.length == test.length)
    for(i <- center.indices){
      error += (center(i)- test(i))*(center(i)- test(i))
    }
    return error
  }


  def predict(center: Array[Array[Double]], test:Array[Array[Double]]):Array[Int] = {
    // test.length row  test(0).length col
    // 如果不使用new来创建那么长度固定为1
    var label = new Array[Int](test.length)

    for(test_index <- test.indices){
      var min = Double.MaxValue
      var temp = 0.0
      for(center_index <- center.indices){
        temp = compute(center(center_index), test(test_index))
        if(min > temp) {
          min = temp
          label(test_index) = center_index
        }
      }
    }
    return label
  }

  def sse(predict_label: Array[Int], center: Array[Array[Double]], test_data: Array[Array[Double]]): Double = {
    var sse_value = 0.0
    for(i <- predict_label.indices) {
      sse_value += compute(test_data(i), center(predict_label(i)))
    }
    return sse_value
  }

  def main(args: Array[String]): Unit = {
    // Creates a Spark context and a SQL context
    val cluster_num = 10

    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("local");
    conf.set("spark.testing.memory", "9147480000");
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val train_path = sc.textFile("E:/App/IDEA/workplace/ml/src/data/save_train_100.txt")
      .filter(_.nonEmpty).map(_.split(" ").map(_.toDouble)).map(Vectors.dense)
    val test_path = sc.textFile("E:/App/IDEA/workplace/ml/src/data/save_train_100.txt")
    val real_label = sc.textFile("E:/App/IDEA/workplace/ml/src/data/save_label_100.txt")
      .filter(_.nonEmpty).map(_.split("\n").map(_.toInt)).collect().flatten

    sc.setLogLevel("ERROR")

    val c1 = train_path.collect()

    val b1 = ArrayBuffer((1,c1(0)))
    for(i <- 1 until train_path.count().toInt ){
      b1 += Tuple2(i+1, c1(i))
    }


    val dataset: DataFrame = sqlContext.createDataFrame(b1).toDF("id", "features")
    val test_data = test_path.filter(_.nonEmpty).map(_.split(" ").map(_.toDouble)).collect()


    for(i_K <- 2 until 30){
      // Trains a k-means model
      var kmeans = new KMeans()
        .setK(i_K)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
      var model = kmeans.fit(dataset)

      var center = model.clusterCenters.map(_.toArray)

      var predict_label = predict(center, test_data)
      var sse_value = sse(predict_label, center, test_data)


      """var counter = 0
      for(index<-0 until cluster_num){
        counter += sc.parallelize((for(i <- real_label.indices) yield if(predict_label(i)==index) real_label(i) else -1).filter(_ != -1)).countByValue.values.max.toInt
      }"""

      // var purity = counter.toDouble/real_label.length.toDouble
      println("-------------------------------------------------------")
      println(s"i_K:$i_K")
      println(s"sse:$sse_value")
      println("-------------------------------------------------------")
      // println(s"Max/Total: $counter / ${real_label.length}")
      // println(s"Purity: $purity")
    }


    sc.stop()

  }
}
// scalastyle:on println