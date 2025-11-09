from metrics.BERTScore import calculate_bertscore, calculate_codebertscore, calculate_finbertscore, calculate_mathbertscore
from metrics.BLEU import calculate_BLEU, calculate_codeBLEU
from metrics.Flesch_Kincaid import calculate_flesch_kincaid
from metrics.Empathy import calculate_empathy
from metrics.Cosine_Similarity import calculate_cosine_similarity
from metrics.Sonar import calculate_sonar_maintainability, calculate_sonar_reliability, calculate_sonar_coverage, calculate_sonar_duplication, calculate_sonar_security
from metrics.Cyclomatic_Complexity import calculate_cyclomatic_complexity
from metrics.FactCC import calculate_factcc
from metrics.Gunning_Fog_Index import calculate_gunning_fog_index
from metrics.BARTScore import calculate_bartscore
from metrics.Length_Ratio import calculate_length_ratio
from metrics.Compiler import calculate_python_compiler, calculate_c_compiler, calculate_java_compiler, calculate_cpp_compiler, calculate_golang_compiler, calculate_js_compiler
from metrics.COMET import calculate_comet
from metrics.ROUGE import calculate_rouge_l, calculate_rouge_n, calculate_rouge_w, calculate_rouge_we_n
from metrics.Perplexity import calculate_perplexity
from metrics.GTM import calculate_gtm
from metrics.NIST import calculate_nist
from metrics.PyLint import calculate_pylint
from metrics.WER import calculate_wer
from metrics.METEOR import calculate_meteor
from metrics.Distinct_N import calculate_distinct_n
from metrics.MOVERSScore import calculate_movers_score
from metrics.Completeness import calculate_completeness
from metrics.Semantic_Analysis import calculate_semantic
from metrics.Grammaly import calculate_grammaly
from metrics.GER import calculate_ger
from metrics.Cognitive_Complexity import calculate_cognitive_complexity
from metrics.GPTScore import calculate_gpt_score
from metrics.Calculator import calculate
from metrics.Z3_solver import calculate_satisfiability
from metrics.MAUVE import calculate_mauve

info = {
    "BERTScore": {
        "description": "Measures semantic similarity between two texts by capturing their contextual, thematic, and emotional alignment using BERT embeddings. The metric evaluates shared concepts, emotional tones, and narrative structures, focusing on deeper semantic relationships beyond exact lexical overlap or structural coherence.",
        "implementation": calculate_bertscore,
        "input": ["context", "response"]
    },
    "codeBERTScore": {
        "description": "Evaluates the similarity of code snippets or programming-related texts using CodeBERT embeddings.",
        "implementation": calculate_codebertscore,
        "input": ["context", "response"]
    },
    "finBERTScore": {
        "description": "Assesses financial text similarity or sentiment using FinBERT embeddings.",
        "implementation": calculate_finbertscore,
        "input": ["context", "response"]
    },
    "mathBERTScore": {
        "description": "Evaluates the quality and relevance of mathematical expressions in natural language processing (NLP) tasks, specifically in mathematical content or equations. Higher MathBERTScore values indicate a closer match between generated and reference expressions, reflecting a better understanding of mathematical semantics and structure.",
        "implementation": calculate_mathbertscore,
        "input": ["context", "response"]
    },
    "BARTScore": {
        "description": "Generates a score for the quality of a response by evaluating its semantic, emotional, and contextual alignment with a given prompt or context using BART embeddings. This metric captures nuanced relationships between the response and context, considering thematic consistency, emotional tone, and logical coherence. Responses with higher scores exhibit stronger contextual relevance, semantic clarity, and structural flow. It is particularly sensitive to deviations from the context, fragmented structures, and disorganized narratives, which result in lower scores.",
        "implementation": calculate_bartscore,
        "input": ["context", "response"]
    },
    "BLEU": {
        "description": "Calculates n-gram overlap between the reference text and the generated response, evaluating lexical and structural similarity. This metric is sensitive to exact word and phrase matches, with higher scores reflecting greater alignment in vocabulary, word order, and phrasing. It is commonly used for machine translation evaluation, where direct lexical correspondence between the source and target texts is a key measure of quality. Synonymy, paraphrasing, or semantic rewording that does not result in shared n-grams is not accounted for, making the metric reliant on explicit textual overlap.",
        "implementation": calculate_BLEU,
        "input": ["reference", "response"]
    },
    "CodeBLEU": {
        "description": "Evaluates the quality of generated code by comparing it to human-written reference code, combining multiple metrics to assess both syntactic accuracy and structural alignment. This composite score balances various aspects of code generation, from preserving syntax to ensuring logical equivalence, prioritizing code that closely matches both the function and structure of the reference. Higher codeBLEU values reflect better overall code generation quality, indicating outputs that are more precise, structurally correct, and semantically aligned with the reference code.",
        "implementation": calculate_codeBLEU,
        "input": ["reference", "response"]
    },
    "Flesch_Kincaid": {
        "description": "Measures the readability of a text based on sentence length and word syllable count, producing a grade-level score indicative of the text's complexity. Longer sentences and multi-syllabic words increase the score, reflecting advanced reading difficulty. This metric focuses on linguistic complexity rather than comprehension or clarity, making it particularly suited for evaluating texts where dense or academic language is prominent.",
        "implementation": calculate_flesch_kincaid,
        "input": ["response"]
    },
    "Empathy": {
        "description": "Evaluates the empathetic tone of a text by analyzing its ability to evoke emotional connection and resonance with readers. This metric considers factors such as emotional clarity, relatability, tone, structure, and the presence of explicit empathy triggers. Texts that effectively express universal emotions, adopt an inviting tone, and guide readers through a relatable emotional journey score higher. Fragmented narratives, abstract themes, or overly internalized emotions may reduce the empathetic impact and result in lower scores.",
        "implementation": calculate_empathy,
        "input": ["response"]
    },
    "Cosine_Similarity": {
        "description": "Computes the cosine similarity between text embeddings to measure semantic alignment or similarity. This metric evaluates the degree to which texts share contextual meaning, thematic coherence, and emotional tone. It is robust to variations in word choice, paraphrasing, and structural differences, focusing instead on underlying ideas and relationships. Higher scores indicate stronger semantic and contextual alignment, while lower scores reflect divergence in meaning or focus.",
        "implementation": calculate_cosine_similarity,
        "input": ["context", "response"]
    },
    "Cognitive_Complexity": {
        "description": "Evaluates the cognitive complexity of source code by analyzing the control flow structures and their nesting levels. This metric calculates the complexity based on the presence of control flow statements (such as if, for, while, switch, etc.) and keywords like break and continue. Each control flow statement contributes to the complexity score, with additional increments for nested blocks, which make the code more challenging to follow. A higher score indicates code that is more difficult to read, understand, and modify, emphasizing the importance of simplicity, clarity, and maintainability in software development.",
        "implementation": calculate_cognitive_complexity,
        "input": ["response"]
    },
    "Sonar_Maintainability": {
        "description": "Analyzes the maintainability of source code based on metrics such as cyclomatic complexity, duplication, readability, and error-proneness using SonarQube. The metric identifies issues like excessive branching, redundant patterns, and poor code clarity, assigning scores that reflect how easy the code is to understand, debug, and modify. Concise code with minimal structural overhead may score better but can sacrifice readability. Verbose implementations with clear logic may incur more issues due to higher complexity or duplication but are often easier to extend and debug. Lower scores indicate fewer issues and higher maintainability.",
        "implementation": calculate_sonar_maintainability,
        "input": ["doc_id", "system_id"]
    },
    "Sonar_Reliability": {
        "description": "Evaluates the reliability of source code by identifying potential bugs, anti-patterns, and problematic practices using SonarQube. This metric penalizes issues such as ambiguous variable names, lack of structure, missing context, and non-adherence to best practices. It prioritizes robust, readable, and maintainable code, assigning issues for patterns that increase error-proneness or reduce clarity. Higher issue counts reflect greater deviations from reliable and maintainable coding standards.",
        "implementation": calculate_sonar_reliability,
        "input": ["doc_id", "system_id"]
    },
    "Sonar_Coverage": {
        "description": "Assesses the extent to which source code is covered by automated tests, using SonarQube's analysis of test execution results. This metric highlights gaps in test coverage, emphasizing the importance of thorough testing for detecting bugs, ensuring code quality, and preventing regressions. It penalizes areas with low or no coverage, which may indicate untested, potentially fragile code. The coverage score prioritizes high-quality code with robust test suites, reflecting the confidence developers can have in the reliability and stability of the application. Higher scores represent more comprehensive test coverage, while lower scores suggest significant areas of the codebase are insufficiently tested.",
        "implementation": calculate_sonar_coverage,
        "input": ["doc_id", "system_id"]
    },
    "Sonar_Duplication": {
        "description": "Evaluates the level of duplicated code within the source codebase using SonarQube. This metric identifies sections of code that are repeated across different parts of the application. It penalizes high levels of code duplication, as it typically indicates poor design practices and reduces code clarity and maintainability. The duplication score prioritizes clean, modular code that minimizes redundancy and maximizes reusability. Higher duplication scores reflect a greater degree of repeated code, signaling areas that could benefit from refactoring or abstraction to improve overall code quality and maintainability.",
        "implementation": calculate_sonar_duplication,
        "input": ["doc_id", "system_id"]
    },
    "Sonar_Security": {
        "description": "Evaluates the security vulnerabilities in source code by identifying potential threats, weaknesses, and insecure coding practices using SonarQube. This metric penalizes issues such as inadequate input validation, improper error handling, exposure of sensitive data, and the use of outdated or insecure libraries. It prioritizes secure coding practices that safeguard against attacks. Higher issue counts indicate a higher potential for security risks, signaling the need for remediation to protect both users and systems.",
        "implementation": calculate_sonar_security,
        "input": ["doc_id", "system_id"]
    },
    "Cyclomatic_Complexity": {
        "description": "Calculates the cyclomatic complexity of source code, representing the number of independent paths through the code.",
        "implementation": calculate_cyclomatic_complexity,
        "input": ["response"]
    },
    "C_Compiler": {
        "description": "Assesses the correctness of C source code by attempting to compile it using the GCC (GNU Compiler Collection) and returning a binary success or failure score. A score of 1 indicates successful compilation, meaning the code is syntactically valid and can be compiled without errors. A score of 0 reflects a compilation failure, signaling that the code contains syntax or structural issues that prevent successful compilation.",
        "implementation": calculate_c_compiler,
        "input": ["response"]
    },
    "C++_Compiler": {
        "description": "Evaluates the correctness of C++ source code by attempting to compile it using the g++ compiler with C++17 standards. A score of 1 indicates that the code successfully compiles, suggesting that it adheres to C++ syntax and structural rules and can be built without errors. A score of 0 reflects a compilation failure, indicating that the code contains syntax errors.",
        "implementation": calculate_cpp_compiler,
        "input": ["response"]
    },
    "Python_Compiler": {
        "description": "Compiles and evaluates the given Python code to check for syntax correctness and errors.",
        "implementation": calculate_python_compiler,
        "input": ["response"]
    },
    "JAVA_Compiler": {
        "description": "Assesses the validity of Java source code by attempting to compile it using the javac compiler. A score of 1 indicates that the code successfully compiles, meaning it adheres to Java's syntax rules and can be built into a class file. A score of 0 signals a compilation failure, indicating that the code contains errors preventing it from compiling correctly.",
        "implementation": calculate_java_compiler,
        "input": ["response"]
    },
    "JavaScript_Compiler": {
        "description": "Assesses the basic validity of JavaScript code before further execution or analysis, ensuring that the code adheres to JavaScript's syntax rules. A score of 1 is returned if the code is successfully parsed without errors, indicating that the JavaScript code is syntactically valid, while a score of 0 is returned if parsing fails, indicating the presence of syntax errors.",
        "implementation": calculate_js_compiler,
        "input": ["response"]
    },
    "Golang_Compiler": {
        "description": "Confirms that Go code is structurally sound and free from errors that would block the build process, ensuring the code is executable. A score of 1 indicates successful compilation, meaning the code is syntactically correct and can be compiled without errors. A score of 0 indicates a compilation failure.",
        "implementation": calculate_golang_compiler,
        "input": ["response"]
    },
    "FactCC": {
        "description": "Checks the factual consistency of a generated response against a given context.",
        "implementation": calculate_factcc,
        "input": ["context", "response"]
    },
    "Gunning_Fog_Index": {
        "description": "Assesses text complexity and readability by analyzing sentence length, word syllable count, and narrative structure, providing a grade-level score that reflects the education level required for comprehension. Longer sentences, multi-syllabic words, and abstract or dense narrative styles contribute to higher complexity scores. This metric is particularly effective for evaluating texts that target advanced readers and is sensitive to linguistic and thematic intricacies that may challenge understanding.",
        "implementation": calculate_gunning_fog_index,
        "input": ["response"]
    },
    "Length_Ratio": {
        "description": "Computes the ratio of lengths between the context and the response to gauge verbosity or conciseness.",
        "implementation": calculate_length_ratio,
        "input": ["context", "response"]
    },
    "COMET": {
        "description": "A metric designed to evaluate machine translation quality based on semantic similarity and contextual understanding using pre-trained models.",
        "implementation": calculate_comet,
        "input": ["context", "response"]
    },
    "ROUGE-L": {
        "description": "Measures the longest common subsequence (LCS) between the context and response, focusing on sentence-level structure and fluency.",
        "implementation": calculate_rouge_l,
        "input": ["reference", "response"]
    },
    "ROUGE-N": {
        "description": "Evaluates the quality of a machine-generated text by measuring the overlap of n-grams (sequences of n words) between the generated content and human-labeled reference text. This metric prioritizes the ability of a model to generate content that accurately reflects the key words and phrases found in the reference text. It penalizes outputs that miss important n-grams or introduce irrelevant terms, while rewarding matches that capture essential elements of the reference. Higher ROUGE-N scores indicate better alignment with the reference text, reflecting the model's ability to reproduce meaningful and relevant n-grams.",
        "implementation": calculate_rouge_n,
        "input": ["response"]
    },
    "ROUGE-W": {
        "description": "Evaluates the quality of text generation by comparing a hypothesis (e.g., model-generated text) to a reference (e.g., human-written text) using a weighted version of the Longest Common Subsequence (LCS). Higher scores reflect better alignment between the reference and hypothesis in terms of content overlap and coherence.",
        "implementation": calculate_rouge_w,
        "input": ["reference", "response"]
    },
    "ROUGE-WE-N": {
        "description": "Measures the quality of a generated text by comparing it to a reference using a weighted Longest Common Subsequence (LCS) with an added focus on token-level n-gram precision. The score is sensitive to both the frequency and order of these n-grams, providing a more detailed and accurate reflection of text quality, especially in tasks like machine translation, summarization, or content generation. Higher scores indicate better alignment in terms of both semantic accuracy and syntactic fluency between the generated text and the reference.",
        "implementation": calculate_rouge_we_n,
        "input": ["reference", "response"]
    },
    "Perplexity": {
        "description": "Evaluates the fluency and predictability of a text by measuring the inverse probability of the response under a language model.",
        "implementation": calculate_perplexity,
        "input": ["response"]
    },
    "GPTScore": {
        "description": "Calculates the negative log-likelihood of the response given the model's learned distribution over the vocabulary, reflecting how probable the model considers the response to be based on its training. A higher GPT score indicates a response that aligns well with the model's expected language patterns and structure, suggesting that the text is fluent, coherent, and contextually appropriate.",
        "implementation": calculate_gpt_score,
        "input": ["response"]
    },
    "WER": {
        "description": "Evaluates the accuracy of speech recognition or text generation by calculating the proportion of word errors between a reference (ground truth) and a hypothesis (generated output). A lower WER indicates that the generated text is closer to the reference, reflecting better performance in tasks like automatic speech recognition (ASR) or machine translation. The metric emphasizes both the preservation of word-level content and the handling of word mismatches, with the final score normalized by the length of the reference.",
        "implementation": calculate_wer,
        "input": ["reference", "response"]
    },
    "GTM": {
        "description": "Measuring the alignment of generated text with human-labeled reference data, using n-gram precision and recall across multiple n-gram sizes. It penalizes discrepancies where the generated text either fails to match reference n-grams (low precision) or misses essential reference n-grams (low recall). A higher GTM score indicates a more balanced, robust model that performs well across various n-gram scales.",
        "implementation": calculate_gtm,
        "input": ["reference", "response"]
    },
    "GER": {
        "description": "Evaluates the proportion of grammatical errors in a given text by comparing it to a corrected version. GER prioritizes accuracy in identifying errors by considering the length and scope of the errors, with a higher score indicating a greater proportion of grammatical issues in the text.",
        "implementation": calculate_ger,
        "input": ["reference", "response"]
    },
    "NIST": {
        "description": "Evaluates the quality of machine-generated text by measuring its alignment with human-labeled reference translations, incorporating both precision and recall in a way that adjusts for the length and fluency of the output. The NIST metric improves upon traditional precision-based metrics by giving more weight to rare, informative n-grams, which are typically more meaningful in evaluating translations or summaries. It penalizes outputs that fail to capture these critical n-grams, prioritizing models that maintain the richness of the reference while balancing brevity. The NIST score also adjusts for over-generation, helping to identify models that produce more fluent but less accurate outputs. Higher NIST scores indicate better overall quality, with a focus on both preserving the important details of the reference and ensuring the generation of content that is contextually relevant and accurate.",
        "implementation": calculate_nist,
        "input": ["reference", "response"]
    },
    "METEOR": {
        "description": "Evaluates the quality of machine-generated text by comparing it to human-labeled reference data, using both exact word matches and synonym matching. The score is computed based on the alignment of words between the source, reference, and candidate text, with penalties for word order mismatches and omitted or extra words. A higher METEOR score indicates that the generated text is closer to the reference in terms of meaning and structure.",
        "implementation": calculate_meteor,
        "input": ["context", "reference", "response"]
    },
    "PyLint": {
        "description": "Evaluates the quality and readability of Python source code by analyzing it for potential issues, coding standard violations, and possible bugs using the Pylint tool. The Pylint score is derived from an analysis of the code’s style, structure, and complexity, with penalties applied for issues such as poor naming conventions, unnecessary complexity, redundant code, and violations of the PEP 8 guidelines. A higher Pylint score indicates cleaner, more maintainable code that aligns with established coding standards and reduces the likelihood of introducing errors.",
        "implementation": calculate_pylint,
        "input": ["response"]
    },
    "Distinct-N": {
        "description": "Evaluates the diversity of n-grams (sequences of n words) in a given set of texts by calculating the ratio of unique n-grams to total n-grams. This score measures the extent to which the text contains a wide variety of distinct word combinations, with a higher score indicating greater lexical diversity.",
        "implementation": calculate_distinct_n,
        "input": ["response"]
    },
    "MOVERSScore": {
        "description": "Evaluates the semantic similarity between two text sequences by calculating the Earth Mover’s Distance (EMD) between their token-level embeddings. The metric takes into account the Euclidean distances between corresponding tokens, weighted by their relevance within the sequence. A lower score indicates higher similarity between the reference and candidate texts, while a higher score signifies greater dissimilarity.",
        "implementation": calculate_movers_score,
        "input": ["context", "response"]
    },
    "Calculator": {
        "description": "Verifies the accuracy of responses to expressions and ensuring that the expected outcomes are achieved. If the result matches the provided response exactly, a score of 1 is returned, indicating correctness. A score of 0 is returned if there is a mismatch between the result and the response, indicating an error in calculation or formatting.",
        "implementation": calculate,
        "input": ["response"]
    },
    "Sentiment Analysis": {
        "description": "Evaluates the emotional tone and sentiment of a text response by measuring the likelihood of the response expressing a positive sentiment. This metric is particularly useful for assessing the emotional alignment of a response, such as in sentiment analysis tasks where understanding whether a response is optimistic, neutral, or negative is important. A higher score suggests a more positively inclined response, while a lower score may indicate a more negative or neutral tone.",
        "implementation": calculate_semantic,
        "input": ["response"]
    },
    "MAUVE": {
        "description": "Evaluates natural language processing models and assessing their ability to produce fluent, coherent, and contextually appropriate text compared to human-written reference texts. A higher MAUVE score indicates that the generated text is more similar to the reference text in terms of content distribution, suggesting better text generation quality.",
        "implementation": calculate_mauve,
        "input": ["reference", "response"]
    },
    "Completeness": {
        "description": "Evaluates the logical coherence and flow of a multi-step response by analyzing the semantic similarity between consecutive steps. A higher average similarity score indicates that the steps in the response are closely related and logically connected, suggesting a more complete and cohesive solution. This metric is particularly useful for evaluating structured problem-solving tasks, such as mathematical problem-solving or step-by-step reasoning.",
        "implementation": calculate_completeness,
        "input": ["response"]
    },
    "Z3_Solver": {
        "description": "Validates logical consistency and solving constraint satisfaction problems. A score of 1 indicates the formula is satisfiable, meaning a solution exists that satisfies all constraints; 0 signifies the formula is unsatisfiable, implying no solution can satisfy the constraints; and -1 represents an error or an unknown result, indicating issues with execution or an indeterminate satisfiability outcome.",
        "implementation": calculate_satisfiability,
        "input": ["response"]
    },
    "Grammarly": {
        "description": "Assesses the grammatical correctness and clarity of written text by identifying potential issues such as spelling errors, incorrect punctuation, awkward sentence structure, and other language inconsistencies. It prioritizes clear, concise, and polished writing, rewarding texts that demonstrate correct usage of syntax, tense, and tone. Higher scores indicate fewer grammar issues and a more refined, professional response.",
        "implementation": calculate_grammaly,
        "input": ["response"]
    }
}