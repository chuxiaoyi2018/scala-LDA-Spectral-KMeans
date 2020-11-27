package main.scala

// scalastyle:off println
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}

object LDA_cluster{

  final val FEATURES_COL = "features"

  def main(args: Array[String]): Unit = {
    val cluster_num = 10

    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("yarn-cluster")
    conf.set("spark.testing.memory", "2147480000");
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val train_path = sc.textFile("/user/2020110336/save_train_5000.txt")
      .filter(_.nonEmpty).map(_.split(" ").map(_.toDouble)).map(Vectors.dense).map(Row(_))
    val test_path = sc.textFile("/user/2020110336/save_test_1000.txt")
      .filter(_.nonEmpty).map(_.split(" ").map(_.toDouble)).map(Vectors.dense).map(Row(_))
    val real_label = sc.textFile("/user/2020110336/test_label_1000.txt")
      .filter(_.nonEmpty).map(_.split("\n").map(_.toInt)).collect().flatten


    val schema = StructType(Array(StructField(FEATURES_COL, new VectorUDT, false)))
    val dataset = sqlContext.createDataFrame(train_path, schema)
    val testset = sqlContext.createDataFrame(test_path, schema)

    // Trains a LDA model
    val lda = new LDA()
      .setK(cluster_num)
      .setMaxIter(10)
      .setFeaturesCol(FEATURES_COL)
    val model = lda.fit(dataset)
    val transformed = model.transform(testset)

    transformed.show(false)
    val col = transformed.select("topicDistribution").collect()

    val predict_label = (for(i<-col.indices) yield col(i).toString()
      .filter(_ != '[').filter(_ != ']').split(",").map(_.toDouble).zipWithIndex.maxBy(_._1)._2).toArray

    var counter = 0
    for(index<-0 until cluster_num){
      counter += sc.parallelize((for(i <- real_label.indices) yield if(predict_label(i)==index) real_label(i) else -1).filter(_ != -1)).countByValue.values.max.toInt
    }

    var purity = counter.toDouble/real_label.length.toDouble
    println(s"Max/Total: $counter / ${real_label.length}")
    println(s"Purity: $purity")


    // $example off$
    sc.stop()
  }
}
// scalastyle:on println
