# testar esse código - https://acadgild.com/blog/apache-spark-hbase

create table 'personal', 'personal_data'

put 'personal', '1000', 'personal_data:city', 'John Dole'
put 'personal', '1000', 'personal_data:name', '1-425-000-0001'
put 'personal', '1000', 'personal_data:city', '1-425-000-0002'
put 'personal', '1000', 'personal_data:name', '1111 San Gabriel Dr.'
put 'personal', '8396', 'personal_data:city', 'Calvin Raji'
put 'personal', '8396', 'personal_data:name', '230-555-0191'
put 'personal', '8396', 'personal_data:city', '230-555-0191'
put 'personal', '8396', 'personal_data:name', '5415 San Gabriel Dr.'
put 'personal',1,'personal_data:name','Ram'

export SPARK_LOCAL_IP=127.0.0.1 # modificação do IP
HBASE_PATH=`/usr/lib/hbase/bin/hbase classpath`

spark-shell --driver-class-path $HBASE_PATH

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.client.HBaseAdmin
import org.apache.hadoop.hbase.{HTableDescriptor,HColumnDescriptor}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.client.{Put,HTable}

val conf = HBaseConfiguration.create()
val tablename = "personal"

conf.set(TableInputFormat.INPUT_TABLE,tablename)
// val admin = new HBaseAdmin(conf)

val sc = new SparkContext(master="local[*]", appName="Spark_Save_Us", new SparkConf)

val HbaseRdd = sc.newAPIHadoopRDD(conf, classOf[TableInputFormat], classOf[org.apache.hadoop.hbase.io.ImmutableBytesWritable], classOf[org.apache.hadoop.hbase.client.Result])

HbaseRdd.count()
