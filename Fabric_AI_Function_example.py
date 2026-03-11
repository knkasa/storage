from synapse.ml.spark.aifunc.DataFrameExtensions import AIFunctions
from synapse.ml.services.openai import OpenAIDefaults

#**** 注意：カラム名は英語でなければならない**********

defaults = OpenAIDefaults()
#defaults.set_deployment_name("gpt-35-turbo-0125")
#defaults.set_deployment_name("gpt-4o-mini")
defaults.set_deployment_name("gpt-4.1-mini")

import os
import pandas as pd
from pyspark.sql import SparkSession

os.chdir('/lakehouse/default/Files')
df = pd.read_csv('survey2024.csv', usecols=['継続意向の理由'])
df = df.rename(columns={'継続意向の理由':'comment'})
df.dropna(inplace=True)

spark = SparkSession.builder.getOrCreate() # you mey not need this.
spark_df = spark.createDataFrame(df)

#**** 注意：カラム名は英語でなければならない**********


# Similarity search
df = pd.DataFrame([ 
        ("Bill Gates"), 
        ("Satya Nadella"), 
        ("Joan of Arc")
    ], columns=["name"])
df["similarity"] = df["name"].ai.similarity("Microsoft")

# Grouping
label_list = ["満足", "不満"]
df_classify = spark_df.ai.classify(labels=label_list, input_col='comment', output_col='category')
display(df_classify)

# Positive/Neutral/Negative
df_sentiment = spark_df.ai.analyze_sentiment(input_col="comment", output_col="sentiment")
display(df_sentiment)

# Entity extraction
df_entities = spark_df.ai.extract(labels=["意見", "日付"], input_col="comment")
display(df_entities)

# Summarize texts
#**** 注意：カラム名は英語でなければならない**********
df_summaries = spark_df.ai.summarize(input_col="comment", output_col="summaries")
display(df_summaries)

# Embedding
df["embed"] = df["descriptions"].ai.embed()

# Original prompts
#**** 注意：カラム名は英語でなければならない**********
#df_responses = spark_df.ai.generate_response(prompt="満足か不満かといえばどっち？", output_col="response")
df_response = spark_df_story.ai.generate_response(
    prompt=(
        "以下のテキストを読み、"
        "『お土産・差し入れ・贈呈・進呈など何らかの贈り物をした』なら 1、"
        "『していない』なら 0 だけを返してください。\n"
        "テキスト: {summary}"
        ),
    is_prompt_template=True,   # <-- tells Fabric to pull the {summary} column for each row
    output_col="response",     # your output column
    )


# ai.fix_grammar()

#Convert to pandas
df_pandas = df_spark.toPandas()
df_pandas.to_parquet(f'file.parquet', engine='pyarrow', index=False)

# Save to parquet.  Use .repartition(4) or .coalesce(1) to save as chunk or single file.
df_event.repartition(4).write.mode("overwrite").parquet("Files/活動メモ_クラスタリング")
df_event = spark.read.parquet("Files/活動メモ_クラスタリング")


#******** Free up memory for spark dataframe. Run Notebook from Pipeline.
# Also, use less tokens, choose GPT mini models.
# https://learn.microsoft.com/en-us/fabric/data-science/ai-functions/pyspark/configuration
del df; gc.collect()
spark_df.unpersist(blocking=True)  # Delete spark dataframe.
spark.catalog.clearCache()  # Clear all spark dataframes.
spark.stop()

#**********Run this before converting from spark to pandas.
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

#============ Save resource usage using AI function with Pandas instead of pyspark. ============================================
# https://blog.fabric.microsoft.com/en-us/blog/introducing-upgrades-to-ai-functions-for-better-performance-and-lower-costs/
# https://learn.microsoft.com/en-us/fabric/data-science/ai-functions/pandas/configuration
# https://github.com/fredgis/fabricaifunctions/blob/main/AIFunctionsDemoFinal.ipynb

# %pip install fabric-ai-functions --quiet
import pandas as pd
import aifunc   # AI functions namespace for pandas
df["ai_output"] = df.ai.generate_response(prompt="Summarize the product description.")

from aifunc import Conf
Conf.concurrency = 10   # number of parallel process records.



#============= Fabric AI Chat Agent ===================================
from pyspark.sql import Row
from pyspark.sql import functions as F
from synapse.ml.services.openai import OpenAIChatCompletion

def msg(role, content):
    return Row(role=role, content=content, name=role)

chat_df = spark.createDataFrame([
    ([ msg("system", "You are a concise assistant."),
       msg("user",   "Summarize why Spark helps with LLM workloads.") ],),
    ([ msg("system", "You respond in Japanese."),
       msg("user",   "Azure OpenAI をバッチ処理で使う利点は？") ],)
]).toDF("messages")


chat = (OpenAIChatCompletion()
        #.setCustomServiceName("<your-azure-openai-resource-name>")   # e.g., aoai-myteam-eastus
        .setDeploymentName("gpt-4.1-mini")                           # your chat model deployment
        #.setSubscriptionKey("<your-azure-openai-key>")               # or setAADToken(...)
        .setMessagesCol("messages")
        .setOutputCol("chat_completions")
        .setErrorCol("error")
        .setMaxTokens(3000)
        .setTemperature(0.5)
        .setConcurrency(8))

# 3) Run at scale and inspect results
result_df = chat.transform(chat_df).select( "messages",F.col("chat_completions.choices.message.content").alias("assistant_reply"), "error")

display(result_df)


