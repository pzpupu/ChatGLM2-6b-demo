from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModel

BASE_MODEL_NAME = 'THUDM/chatglm2-6b'
PEFT_MODEL_NAME = "pzpupu/chatglm2-6b-lora_version"
QUESTION = "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳"


# 测试模型没有进行Finetune之前
def test_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, device_map="auto").half()
    model = model.eval()
    response, history = model.chat(tokenizer, QUESTION)
    print(response)


# 测试Finetune之后的模型
def test_finetune_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(PEFT_MODEL_NAME, trust_remote_code=True, device_map="auto").half()
    model = model.eval()
    response, history = model.chat(tokenizer, QUESTION)
    print(response)


if __name__ == '__main__':
    test_model()
    # 根据题目中给出的信息，我们可以得到以下答案：
    #
    # 类型：宽松
    #
    # 版型：显瘦
    #
    # 图案：线条
    #
    # 衣样式：衬衫
    #
    # 衣袖型：泡泡袖
    #
    # 衣款式：抽绳
    #
    # 因此，这些信息描述了上衣的宽松版型，以显瘦为目标，采用泡泡袖和抽绳设计的衬衫款式。

    test_finetune_model()
    # 这款衬衫采用了宽松的版型，打造出显瘦的视觉效果，同时也增加了整体造型的层次感。衬衫的泡泡袖设计，既增加了整体造型的层次感，也显得很有活力。抽绳的设计，不仅方便调节袖口，也可以增加整体造型的层次感。
