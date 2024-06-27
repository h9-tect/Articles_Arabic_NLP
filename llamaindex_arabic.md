# دليل شامل لـ LlamaIndex في عام 2024

## جدول المحتويات
1. [فهم LlamaIndex](#فهم-llamaindex)
2. [إنشاء Llamaindex Documents](#إنشاء-llamaindex-documents)
3. [إنشاء LlamaIndex Nodes](#إنشاء-llamaindex-nodes)
4. [إنشاء LlamaIndex Index](#إنشاء-llamaindex-index)
5. [تخزين Index](#تخزين-index)
6. [استخدام Index للاستعلام عن البيانات](#استخدام-index-للاستعلام-عن-البيانات)
7. [المخرجات المنظمة](#المخرجات-المنظمة)
8. [استخدام Index للدردشة مع البيانات](#استخدام-index-للدردشة-مع-البيانات)
9. [أدوات Llamaindex وعملاء البيانات](#أدوات-llamaindex-وعملاء-البيانات)
10. [أمثلة إضافية وحالات استخدام](#أمثلة-إضافية-وحالات-استخدام)

## فهم LlamaIndex

LlamaIndex هو إطار عمل قوي ومرن مصمم لتبسيط عملية بناء تطبيقات الذكاء الاصطناعي مع إمكانية الوصول إلى البيانات الخاصة. يعمل كجسر بين نماذج اللغة الكبيرة (LLMs) ومصادر البيانات الخارجية، مما يتيح إنشاء تطبيقات ذكية قادرة على الاستفادة من المعلومات الخاصة والحديثة.

الميزات الرئيسية لـ LlamaIndex:
1. **Data Ingestion**: يوفر آليات لاستيراد البيانات من مصادر متنوعة مثل الملفات النصية و PDFs و JSONs و APIs.
2. **Indexing**: ينشئ فهارس فعالة للبيانات، مما يسهل استرجاع المعلومات ذات الصلة بسرعة.
3. **Querying**: يسمح بالاستعلام عن البيانات المفهرسة باستخدام اللغة الطبيعية.
4. **LLM Integration**: يتكامل بسلاسة مع نماذج اللغة الكبيرة مثل GPT-3 و GPT-4.

## إنشاء Llamaindex Documents

تعد Data connectors، المعروفة أيضًا باسم Readers، مكونات حاسمة في LlamaIndex تسهل استيراد البيانات من مصادر وتنسيقات مختلفة، وتحويلها إلى تمثيل Document مبسط.

### مثال: تحميل ملف PDF

```python
from llama_index import SimpleDirectoryReader

# تحميل ملف PDF
loader = SimpleDirectoryReader(input_files=["path/to/your/pdf/file.pdf"])
documents = loader.load_data()
```

### مثال: تحميل صفحة Wikipedia

```python
from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['New York City', 'Los Angeles', 'Chicago'])
```

## إنشاء LlamaIndex Nodes

بعد استيراد البيانات كـ Documents، يسمح LlamaIndex بمزيد من المعالجة إلى Nodes، وهي كيانات بيانات أكثر دقة تمثل "أجزاء" من Documents المصدر.

### إنشاء Node أساسي

```python
from llama_index.node_parser import SimpleNodeParser

# بافتراض أنه تم تحميل المستندات بالفعل
parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)
```

### إنشاء Node متقدم

```python
from llama_index.text_splitter import SentenceSplitter
import tiktoken

text_splitter = SentenceSplitter(
  separator=" ", chunk_size=1024, chunk_overlap=20,
  paragraph_separator="\n\n\n", secondary_chunking_regex="[^,.;。]+[,.;。]?",
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
```

## إنشاء LlamaIndex Index

الفهرسة هي وظيفة أساسية في LlamaIndex، تسمح بالاستعلام الفعال عن البيانات المستوردة.

### مثال: إنشاء VectorStoreIndex

```python
from llama_index import VectorStoreIndex

# بافتراض أن docs هي قائمة كائنات Document الخاصة بك
index = VectorStoreIndex.from_documents(docs)
```

### مثال: إنشاء Summary Index

```python
from llama_index import SummaryIndex

# بافتراض أن nodes هي قائمة كائنات Node الخاصة بك
index = SummaryIndex(nodes)
```

## تخزين Index

يوفر LlamaIndex إمكانيات لإدارة تخزين البيانات، بما في ذلك ميزات التخصيص والاستمرارية.

### الاستمرارية الأساسية

```python
# الحفظ على القرص
index.storage_context.persist(persist_dir="<persist_dir>")

# التحميل من القرص
from llama_index import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
index = load_index_from_storage(storage_context)
```

### تكوين التخزين المتقدم

```python
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage import StorageContext

storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore(),
    vector_store=SimpleVectorStore(),
    index_store=SimpleIndexStore(),
)
```

## استخدام Index للاستعلام عن البيانات

بعد إنشاء فهرس منظم جيدًا، الخطوة التالية هي الاستعلام عن هذا الفهرس لاستخراج رؤى ذات معنى.

### واجهة برمجة الاستعلامات عالية المستوى

```python
# بافتراض أن 'index' هو كائن الفهرس المنشأ الخاص بك
query_engine = index.as_query_engine()
response = query_engine.query("ما هي أهم الأحداث التاريخية في نيويورك؟")
print(response)
```

### واجهة برمجة التكوين منخفضة المستوى

```python
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

# بناء الفهرس وتكوين المسترجع
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)

# تكوين مولد الاستجابة
response_synthesizer = get_response_synthesizer()

# تجميع محرك الاستعلام مع معالجات ما بعد العملية
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
)

# تنفيذ الاستعلام
response = query_engine.query("ما هي أشهر المعالم السياحية في لوس أنجلوس؟")
print(response)
```

## المخرجات المنظمة

يمكن لـ LlamaIndex الاستفادة من قدرات نماذج اللغة الكبيرة (LLMs) لتقديم نتائج منظمة.

### الطريقة 1: برامج Pydantic

```python
from pydantic import BaseModel, Field
from typing import List

class CityInfo(BaseModel):
    city: str = Field(description="اسم المدينة")
    population: int = Field(description="عدد سكان المدينة")
    landmarks: List[str] = Field(description="أهم المعالم السياحية في المدينة")

from llama_index.program import OpenAIPydanticProgram

city_info_program = OpenAIPydanticProgram.from_defaults(
    output_cls=CityInfo,
    prompt_template_str="بالنظر إلى معلومات السياق حول {city}، استخرج المعلومات التالية:\n{format_instructions}\n",
    verbose=True,
)

query_engine = index.as_query_engine(
    response_synthesizer=city_info_program
)

# استعلام للحصول على معلومات منظمة
response = query_engine.query("أخبرني عن مدينة نيويورك")
print(response)
```

### الطريقة 2: محللات الإخراج

```python
from llama_index.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="city", description="اسم المدينة"),
    ResponseSchema(name="population", description="عدد سكان المدينة"),
    ResponseSchema(name="landmarks", description="أهم المعالم السياحية في المدينة"),
]

output_parser = StructuredOutputParser(response_schemas=response_schemas)
lc_output_parser = LangchainOutputParser(output_parser)

query_engine = index.as_query_engine(
    output_parser=lc_output_parser,
)

response = query_engine.query("أخبرني عن مدينة لوس أنجلوس")
print(response)
```

## استخدام Index للدردشة مع البيانات

يقدم LlamaIndex مفهوم Chat Engine لتسهيل حوار أكثر تفاعلية وسياقية مع بياناتك.

### الاستخدام الأساسي لمحرك الدردشة

```python
# بناء محرك دردشة من الفهرس
chat_engine = index.as_chat_engine()

# بدء محادثة
response = chat_engine.chat("ما هي أهم المدن في الولايات المتحدة الأمريكية؟")
print(response)

# للاستجابات المتدفقة
streaming_response = chat_engine.stream_chat("قارن بين نيويورك ولوس أنجلوس من حيث السكان والثقافة.")
for token in streaming_response.response_gen:
    print(token, end="")
```

### تكوين أوضاع الدردشة المختلفة

```python
# وضع ReAct Agent
chat_engine = index.as_chat_engine(chat_mode="react", verbose=True)

# وضع OpenAI Agent
from llama_index.llms import OpenAI
service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613"))
chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

# وضع Context
from llama_index.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="أنت خبير في المدن الأمريكية. يمكنك تقديم معلومات مفصلة عن التاريخ والثقافة والسياحة في المدن الكبرى."
)

# وضع Condense Question
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
```

## أدوات Llamaindex وعملاء البيانات

تأخذ LlamaIndex Data Agents اللغة الطبيعية كمدخل وتنفذ إجراءات بدلاً من إنشاء استجابات.

### مثال: استخدام OpenAI Function API-based Data Agent

```python
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata

# إنشاء أدوات للمدن المختلفة
nyc_tool = QueryEngineTool(
    query_engine=nyc_index.as_query_engine(),
    metadata=ToolMetadata(
        name="nyc_query_engine",
        description="يوفر معلومات عن مدينة نيويورك"
    )
)

la_tool = QueryEngineTool(
    query_engine=la_index.as_query_engine(),
    metadata=ToolMetadata(
        name="la_query_engine",
        description="يوفر معلومات عن مدينة لوس أنجلوس"
    )
)

tools = [nyc_tool, la_tool]

# إنشاء نموذج OpenAI
llm = OpenAI(model="gpt-3.5-turbo-0613")

# إنشاء العميل (Agent) باستخدام الأدوات ونموذج اللغة
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

# استخدام العميل للإجابة على الأسئلة
response = agent.chat("قارن بين نيويورك ولوس أنجلوس من حيث عدد السكان والمناخ.")
print(response)

# يمكنك طرح المزيد من الأسئلة
response = agent.chat("ما هي أشهر المعالم السياحية في نيويورك؟")
print(response)

response = agent.chat("كيف يختلف الطقس في لوس أنجلوس عن نيويورك؟")
print(response)
```

## أدوات Llamaindex وعملاء البيانات (تكملة)

```python
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata

# إنشاء أدوات للمدن المختلفة
nyc_tool = QueryEngineTool(
    query_engine=nyc_index.as_query_engine(),
    metadata=ToolMetadata(
        name="nyc_query_engine",
        description="يوفر معلومات عن مدينة نيويورك"
    )
)

la_tool = QueryEngineTool(
    query_engine=la_index.as_query_engine(),
    metadata=ToolMetadata(
        name="la_query_engine",
        description="يوفر معلومات عن مدينة لوس أنجلوس"
    )
)

tools = [nyc_tool, la_tool]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

# الآن يمكنك التفاعل مع العميل
response = agent.chat("قارن بين نيويورك ولوس أنجلوس من حيث عدد السكان والمناخ.")
print(response)
```

في هذا المثال، قمنا بإنشاء أدوات منفصلة لكل مدينة، مما يتيح للوكيل الوصول إلى معلومات محددة عن كل مدينة عند الحاجة.

## أمثلة إضافية وحالات استخدام

### 1. تحليل المستندات المالية

يمكن استخدام LlamaIndex لتحليل التقارير المالية وإستخراج المعلومات الهامة:

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import OpenAI

# تحميل التقارير المالية
documents = SimpleDirectoryReader('financial_reports/').load_data()

# إنشاء الفهرس
index = VectorStoreIndex.from_documents(documents)

# إنشاء محرك الاستعلام
query_engine = index.as_query_engine()

# طرح أسئلة حول البيانات المالية
response = query_engine.query("ما هو إجمالي الإيرادات للربع الأخير؟")
print(response)
```

### 2. مساعد الدردشة للدعم الفني

يمكن استخدام LlamaIndex لإنشاء مساعد دردشة للدعم الفني:

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.memory import ChatMemoryBuffer

# تحميل وثائق الدعم الفني
documents = SimpleDirectoryReader('tech_support_docs/').load_data()
index = VectorStoreIndex.from_documents(documents)

# إعداد ذاكرة الدردشة
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# إنشاء محرك الدردشة
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory,
    system_prompt="أنت مساعد دعم فني ودود ومفيد. استخدم المعلومات المتاحة للإجابة على أسئلة العملاء."
)

# بدء الدردشة
while True:
    user_input = input("العميل: ")
    if user_input.lower() == 'خروج':
        break
    response = chat_engine.chat(user_input)
    print(f"المساعد: {response}")
```

### 3. تلخيص المقالات الإخبارية

يمكن استخدام LlamaIndex لتلخيص المقالات الإخبارية:

```python
from llama_index import Document, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI

# إنشاء مستند من مقالة إخبارية
article_text = """
[نص المقالة الإخبارية هنا]
"""
document = Document(text=article_text)

# تقسيم المستند إلى عقد
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents([document])

# إنشاء الفهرس
index = VectorStoreIndex(nodes)

# إنشاء محرك الاستعلام مع تخصيص LLM
llm = OpenAI(model="gpt-3.5-turbo-16k")
query_engine = index.as_query_engine(llm=llm)

# طلب تلخيص
summary = query_engine.query("لخص هذه المقالة الإخبارية في خمس نقاط رئيسية.")
print(summary)
```

## الخاتمة

يوفر LlamaIndex مجموعة قوية من الأدوات لبناء تطبيقات ذكية تستفيد من قوة نماذج اللغة الكبيرة مع البيانات الخاصة. من خلال الأمثلة والشروحات المقدمة في هذا الدليل، يمكنك البدء في استخدام LlamaIndex لمجموعة متنوعة من التطبيقات، بدءًا من أنظمة الأسئلة والأجوبة البسيطة وحتى المساعدين الافتراضيين المتقدمين وأدوات تحليل البيانات.

تذكر أن هذا الدليل يغطي فقط الأساسيات، وهناك المزيد من الميزات والتخصيصات المتقدمة المتاحة في LlamaIndex. نشجعك على استكشاف الوثائق الرسمية وتجربة الميزات المختلفة لتحقيق أقصى استفادة من هذه الأداة القوية في مشاريعك.

للمزيد من المعلومات والتحديثات، يرجى زيارة [الموقع الرسمي لـ LlamaIndex](https://www.llamaindex.ai/).
