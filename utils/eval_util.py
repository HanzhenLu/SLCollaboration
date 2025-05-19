import re
import timeout_decorator
import json
import tree_sitter_python as tspython
from nltk.tokenize import RegexpTokenizer
from tree_sitter import Language, Parser

IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''

code_tokenizer = RegexpTokenizer(r'\w+')

parser = None

@timeout_decorator.timeout(5)
def get_ast(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, "utf8")
    try:
        tree = parser.parse(code)
        return tree
    except Exception as e:
        return None

def is_parse_valid(parser, code):
    def syntax_error(node):
        if node.type == "ERROR":
            return True
        try:
            for child in node.children:
                if syntax_error(child):
                    return True
        except RecursionError as err:
            return True

        return False

    tree = get_ast(parser, code)
    if tree is not None:
        return not syntax_error(tree.root_node)
    return False

# def get_python_one_statement(prompt, completion: str, parser):
#     for i in range(len(completion)):
#         code = prompt + completion[:i + 1]
#         if not is_parse_valid(parser, code):
#             continue

#         if i + 1 < len(completion) and completion[i + 1] == "\n":
#             return completion[:i + 1].rstrip()

#     return completion
def get_python_one_statement(prompt: str, completion: str, parser):
    lines = completion.splitlines(keepends=True)  # 保留换行符
    buffer = ''

    for _, line in enumerate(lines):
        buffer += line
        code = prompt + buffer

        if not is_parse_valid(parser, code):
            continue

        # 当前已完成一行，且语法合法
        return buffer.rstrip()

    return completion.rstrip()


def postprocess_code_lines(prompt, completion, parser, lang):
    try:
        if lang == "python":
            return get_python_one_statement(prompt, completion, parser)
        # if lang in ["java", "csharp", "typescript"]:
        #     return get_bracket_lang_statement(completion)
    except Exception as e:
        return completion

def remove_comments(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    return code

def process_example_inline(lang, parser, pred, groundtruth, prefix):
    prediction = postprocess_code_lines(prefix,pred,parser,lang)
    target = groundtruth
    target = remove_comments(target)
    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    # em_label = int(pred_lines == gt_lines)
    return prediction, pred_lines == gt_lines

def process_examples(lang,parser, args):
    sample = args

    prediction = postprocess_code_lines(sample["prefix"], sample["pred"], parser, lang)

    target = sample["ground_truth"]

    target = remove_comments(target)
    pred_lines = [l.strip().lower() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip().lower() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)

    trunc_s = {
        "task_id": sample["task_id"],
        "pred": prediction,
        "target": target,
        "time":sample['time']
    }
    return trunc_s, em_label

