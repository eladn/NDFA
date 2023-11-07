# NDFA (Neural Data-Flow Analysis)
A combination of various neural code models for tackling programming-related tasks (like LogVar, function name prediction, and VarMisUse) using deep neural networks. The principal implementation is the *Hierarchic* model, that is designed especially for tasks involving data-flow-alike information-propagation requirements. We also provide a full implementation for [`code2seq`](https://github.com/tech-srl/code2seq) by Alon et al., and a partial implementation for [*Learning to Represent Programs with Graphs*](https://miltos.allamanis.com/publications/2018learning/) By Allamanis et al.

## Table of contents
- [Intro: Hierarchical code representation](#intro-hierarchical-code-representation)
- [Method encoding types](#method-encoding-types)
- [Quick start](#quick-start)
- [Execution parameters structure (and model's hyper-parameters)](#execution-parameters-structure-and-models-hyper-parameters)
- [Advanced execution options](#advanced-execution-options)
- [Preprocessed data](#preprocessed-data)
- [Expand-Collapse path-based graph encoding framework](#expand-collapse-path-based-graph-encoding-framework)
- [Advanced scattered operators](#advanced-scattered-operators)

## Intro: Hierarchical code representation
This project consists the full implementation of the [*Hierarchic Code Encoder*](https://bit.ly/3vgzclc) 
[[*slides*](https://bit.ly/3ttBtac)] in `PyTorch`. The idea of the hierarchic model is to (i) shorten distances between 
related procedure's elements for effective information-propagation; (ii) utilize the effective paths-based 
information-propagation approach; (iii) scalability - efficiently support lengthy procedure; and (iv) provide a 
fine-grained exploration framework to identify the relevant elements/relations in the code's underlying structure that 
most benefit the model for the overall task.

The procedure's code is being broken at the statement level to form the two-level hierarchic structure. The upper level 
consists control structures (loops, if-stmts, block-stmts), while the lower level consists of expression-statements / 
conditions. In fact, each statement in the lower-level is associated to a node in the procedure's statement-level CFG 
(control-flow graph).
![Upper-Lower AST Hierarchic Leveling Figure](https://gitfront.io/r/user-5760758/c3d4b342f8f48fe8e3764b2c30925ea140b99535/NDFA/raw/doc/figures/upper-lower-ast-split-figure.webp "Upper-Lower AST Hierarchic Leveling Figure")

The hierarchic encoder first applies a *micro* operator to obtain a local encoding for each individual statement (CFG 
node), then it employs a *macro* operator to propagate information globally (between CFG nodes), then it updates the 
local encodings by mixing it with the *globally-aware* encodings of the related CFG nodes, and finally it employs the 
micro operator once again. The *micro* operator is the *local* independent encoder of the statements (statements are 
being encoded stand-alone in the first stage). We supply multiple options for the micro operator, including the 
followings: paths-based AST (w/wo *collapse* stage), TreeLSTM AST, GNN AST, AST leaves, flat tokens sequence. The 
*macro* operator is responsible for the *global* information propagation between CFG nodes. We supply the following 
macro operators: paths-based CFG, GNN over CFG, upper control AST (paths-based, GNN, TreeLSTM, leaves only), single 
sequence of CFG nodes (ordered by textual appearance in code), set of CFG nodes.
![Hierarchic Framework Overview Figure](https://gitfront.io/r/user-5760758/c3d4b342f8f48fe8e3764b2c30925ea140b99535/NDFA/raw/doc/figures/hierarchic-framework-overview-figure.webp "Hierarchic Framework Overview Figure")

TODO: state that the thesis shows that different representations (and corresponding encoders) are preferred for different levels.

## Method encoding types
TODO: present the different code encoders with illustrations

### Global (non-hierarchical) non-parsed method code as sequential text
TODO

### Global (non-hierarchical) method AST
TODO
as set-of-paths (using attention over sequences), as a tree (using TreeLSTM), or as a general graph (using GNN).

### Global (non-hierarchical) method AST w/ control edges
TODO
as set-of-paths or as general graph (using GNN). 

### Hierarchical
TODO
#### Micro (local) encoder
The operator that encodes the expression-statements locally.
TODO
#### Macro (global) encoder
The operator that propagates information globally across the different expression-statements. This stage treats each 
expression-statement atomically; that is, it isn't aware of the internal structure of each expression (already encoded 
by the micro encoder). This encoder is aware of the control-flow of the method. 

### Identifiers encoder
All models allow encoding the identifiers (variable, field and type names) separately. So that whenever an identifier 
is attributed with a token/node, its encoded vector is injected to the corresponding token/node embedding. We support 
various manners for encoding identifiers.
See more details in [IdentifierEncoderParams](ndfa\code_nn_modules\params\identifier_encoder_params.py).
To specify the relevant parameters use the arguments prefix `--expr.hp.code-encoder.identifier_encoder`.

## Quick start
We describe how to execute the entire flow in 5 stages: clone & create conda env, download raw / scrape, extract, 
preprocess, train & evaluate.

### Step 1: Clone NDFA repo and set-up conda environment
```bash
>> git clone https://github.com/eladn/NDFA.git
>> cd NDFA
>> conda env create -f environment.[cpu/cuda].yml
>> conda activate NDFA
```

### Step 2 (alternative A): Download raw textual code dataset
```bash
>> mkdir -p data/raw_textual_code_data
>> cd data/raw_textual_code_data
>> wget https://s3.amazonaws.com/code2vec/data/java14[s/m/l]_data.tar.gz
>> tar -xzvf java14m_data.tar.gz
>> cd ../../
```

### Step 2 (alternative B): Scrape `GitHub` for custom dataset creation
Instead of using our datasets, you can use our [GitHub scraper tool](https://github.com/eladn/github_scraper) to form 
your own dataset. This allows both creating a fresh dataset from up-to-date open-source projects. Additionally, it also 
allows customizing the data sources, topics, size, popularity and quality (for more info 
[read here](https://github.com/eladn/github_scraper/blob/main/github_repository_popularity.py)).

### Step 3: Extract parsed AST & CFG structures from textual code
Follow steps from 
[our Java extraction tool repo](https://github.com/eladn/logging-research/tree/master/LoggingsExtractor/JavaExtractor).
This parses the textual java code files into abstract-syntax-trees (AST) and control-flow-graphs (CFG) and serialized 
as json files following [this format interface](ndfa/misc/code_data_structure_api.py). 

### Step 4: Preprocess
```bash
>> python ndfa.py --preprocess \
       --pp-data ./data/pp_data \
       --pp_storage_method rocksdb \
       --raw-train-data-path ./data/raw_extracted_data/train \
       --raw-validation-data-path ./data/raw_extracted_data/val
```

### Step 5: Train & evaluate
```bash
>> python ndfa.py --train --eval \
       --model-save-path ./trained_models \
       --pp-data ./data/pp_data \
       --pp_storage_method rocksdb \
       --batch-size 32 \
       --expr.trn.eff_batch_size 64 \
       --expr.trn.nr_epochs 20 \
       --expr.trn.optimizer AdamW \
       --expr.trn.gradient_clip 0.5 \
       --expr.trn.weight_decay 1e-3 \
       --expr.trn.learning_rate_decay 0.02 \
       --expr.trn.reduce_lr_on_plateau \
       --expr.task.name pred-log-vars \
       --expr.dataset.name java_med \
       --expr.hp.activation_fn leaky_relu \
       --expr.hp.code-encoder.method_encoder_type Hierarchic \
       --expr.hp.code-encoder.hierarchic_encoder.local_expression_encoder.encoder_type AST \
       --expr.hp.code-encoder.hierarchic_encoder.local_expression_encoder.ast_encoder.encoder_type PathsFolded \
       --expr.hp.code-encoder.hierarchic_encoder.local_expression_encoder.ast_encoder.paths_sequence_encoder_params.encoder_type rnn \
       --expr.hp.code-encoder.hierarchic_encoder.local_expression_encoder.ast_encoder.paths_sequence_encoder_params.encoder_type.rnn_type lstm \
       --expr.hp.code-encoder.hierarchic_encoder.local_expression_encoder.ast_encoder.path_sequence_encoder leaf_to_leaf, leaf_to_root, siblings_w_parent_sequences, leaves_sequence \
       --expr.hp.code-encoder.hierarchic_encoder.global_context_encoder.encoder_type CFGPaths \
       --expr.hp.code-encoder.hierarchic_encoder.global_context_encoder.paths_encoder.output_type FoldNodeOccurrencesToNodeEncodings \
       --expr.hp.code-encoder.hierarchic_encoder.global_context_encoder.paths_encoder.path_sequence_encoder.encoder_type transformer
```

## Execution parameters structure (and model's hyper-parameters)
The entire set of parameters for the execution is formed as a nested classes structure rooted at the class 
[`ExecutionParameters`](ndfa/execution_parameters.py). The [`ExecutionParameters`](ndfa/execution_parameters.py) includes the [`ExperimentSetting`](ndfa/experiment_setting.py), which includes 
the [`CodeTaskProperties`](ndfa/code_tasks/code_task_properties.py), the [`NDFAModelHyperParams`](ndfa/ndfa_model_hyper_parameters.py), the [`NDFAModelTrainingHyperParams`](ndfa/ndfa_model_hyper_parameters.py), and the 
[`DatasetProperties`](ndfa/nn_utils/model_wrapper/dataset_properties.py).

Below is a partial example of the higher-level parameters classes definitions to demonstrate this structure:
```python
class ExperimentSetting:
    model_hyper_params: NDFAModelHyperParams = ...
    task: CodeTaskProperties = ...
    train_hyper_params: NDFAModelTrainingHyperParams = ...
    dataset: DatasetProperties = ...

class NDFAModelHyperParams:
    method_code_encoder: MethodCodeEncoderParams = ...
    ...

class MethodCodeEncoderParams:
    method_encoder_type: EncoderType = ...
    hierarchic_micro_macro_encoder: Optional[HierarchicMicroMacroMethodCodeEncoderParams] = ...
    whole_method_expression_encoder: Optional[CodeExpressionEncoderParams] = ...
    method_cfg_encoder: Optional[MethodCFGEncoderParams] = ...
    ...
```

Because this project practically covers many different models, the class [`NDFAModelHyperParams`](ndfa/ndfa_model_hyper_parameters.py) and its 
sub-classes have many optional and choice fields to choose the concrete model to be used. The relevant modules are 
loaded dynamically upon execution according to these parameters.

Generally, each module has its own parameters class that is used wherever this model is needed. For example, the module 
[`CFGPathsMacroEncoder`](ndfa/code_nn_modules/cfg_paths_macro_encoder.py) uses the dedicated parameters class [`CFGPathsMacroEncoderParams`](ndfa/code_nn_modules/params/cfg_paths_macro_encoder_params.py), while the latter 
has a field of class [`SequenceEncoderParams`](ndfa/nn_utils/modules/params/sequence_encoder_params.py) because [`CFGPathsMacroEncoder`](ndfa/code_nn_modules/cfg_paths_macro_encoder.py) uses the module 
[`SequenceEncoder`](ndfa/nn_utils/modules/sequence_encoder.py) accordingly. That is, two parallel trees are constructed - one is the tree of parameters 
(which is constructed independently to the second tree), and the other is the practical modules tree (that is 
constructed w.r.t the params tree).

The main executable script is [`ndfa.py`](ndfa.py). The parameters can be specified as arguments to the script or as a 
`yaml` file. If a trained model is being loaded, its hyper-parameters [`NDFAModelHyperParams`](ndfa/ndfa_model_hyper_parameters.py) are being loaded as 
well and the given ones (through arguments / yaml file) would be ignored. 

After parsing the input config from all input sources (yaml, arguments, defaults), it is being sanitized into a canonic 
representation (remove redundant non-used fields), so that same semantic config has a deterministic config structure 
and deterministic hash. This alleviates experiments re-production, training resumption, and re-usage of compatible 
previously generated preprocessed data.

## Advanced execution options

### Tracking with `W&B`
Create a text file named `wandb_token.txt` within the `credentials` directory (with similar format as 
[`credentials/wandb_token.txt.example`](credentials/wandb_token.txt.example)). It should contain a valid W&B token to 
be used for logging in. Then, to enable, run the main script with the flag `--use-wandb-logger`. 

### Google drive
The training script supports an option to store all training artifacts into Google Drive. To use this option, firstly 
obtain a token to access google drive in the format of 
[`credentials/gdrive_credentials.json.example`](credentials/wandb_token.txt.example). Then, store the actual obtained 
credentials as a text file named `gdrive_credentials.json` within the `credentials` directory. Then, to enable, run the 
main script with the flag `--use-gdrive-logger`.

### Using `notify.run`
It is possible to use [`notify.run`](https://notify.run/) framework to track the training progress remotely via phone / desktop. To enable, run the main script with the flag `--use-notify`. In the beginning of the execution a fresh `notify.run` link will be created for this execution and printed to stdout. 

### Using `Jenkins` for experiments triggering
In our [Jenkins files repo](https://github.com/eladn/ml4code-jenkinsfiles) we created a dedicated `Jenkinsfile` that 
defines an adhoc Jenkins job for each computation flow (code extraction, preprocess, train, extract statistics, etc..). 
That is, it exports all the execution parameters & model hyper-parameters as Jenkins parameters (provided to the user 
as UI textboxes / dropboxes in the build creation form). 
It creates a clean execution environment for each triggered experiment. It automatically fills out all the "boilerplate"
arguments for running the experiment (like paths). It can use different machines for performing the preprocess (CPU 
heavy task) and for performing the training (GPU heavy task) and automatically connects to the relevant node via ssh, 
executes whatever necessary there, copies the data between nodes and maintains a local cache of preprocessed data in 
the nodes. Additionally, it automatically checks for availability and assigns instance for job. It can connect to AWS 
instance (using aws CLI) or to any other private instance. For example, 
[here](https://github.com/eladn/ml4code-jenkinsfiles/blob/main/logvar-ndfa-preprocess/Jenkinsfile) is the preprocessing 
Jenkinsfile and [here](https://github.com/eladn/ml4code-jenkinsfiles/blob/main/logvar-ndfa-train/Jenkinsfile) is the 
training Jenkinsfile. To use it, one should create a Jenkins instance, define its jobs to get their configuration from 
this repo, and define the relevant environment variables in the Jenkins dashboard with the tokens and instances 
information. 

## Preprocessed data
As this projects covers implementation of a variety of different models, each model requires different preprocessed 
data. However, in our implementation, the single class [`MethodCodeInputTensors`](ndfa/code_nn_modules/code_task_input.py) and its sub-classes cover all 
the necessary preprocessed tensors to support all possible models. Practically, if the entire preprocessed data (union 
of all fields needed for all possible models) were to be used, the data loading would become a bottleneck for the 
training/evaluating process. Thus, we allow most of the fields to be optional and we supply the functionality to emit 
the dedicated preprocessed dataset that matches a chosen model. Unless `--keep_entire_preprocessed_dataset` is stated, 
only the tensor fields that are relevant to the given model parameters are stored in the output preprocessed data. For 
training a model, unless `--no-use_compatible_pp_data_if_exists` is set, the data loader seeks for a compatible 
preprocessed data. It uses the lighter preprocessed dataset out of all datasets that comprises the required fields for  
the chosen model.

The preprocessed dataset is stored as a persistent key-value data-structure, where the key is a simple index in the 
range [0..|dataset|-1]. This allows random-access to examples by their index, as necessary for shuffling the data 
while training. We use facebook's `RocksDB` embedded database as our default (and prefered) key-value store backend, 
while we also support other options like `dbm` and archives (zip, tar) possibly with compressions. `RocksDB` shown 
the best random-access read performance for our workload.

### Preprocessed sample input tensors organization and indexing-aware batching
A [preprocessed data sample](ndfa/code_nn_modules/code_task_input.py#L121) is stored as a nested data classes 
structure, where each field is a tensor, a python's primitive (numeric/str), or an instance of another such data class. 
All the classes in this hierarchy inherit from [`TensorsDataClass`](ndfa/misc/tensors_data_class) to support advanced 
batching functionalities for flattened data.

Our input data is formed of several types of elements; that is, each pre-processed example contains multiple kinds of 
entities: CFG nodes, AST nodes, tokens, symbols, identifiers, and paths. Some of the hierarchic method's calculations 
involve elements of multiple kinds, and an element can participate in multiple calculations. We typically use the index 
of an element to address it in the computation. For example, constructing the local encodings of a CFG node requires 
combining all AST paths of the top-level expression sub-AST associated with it. To do so, we store dedicated mappings 
from the CFG node indices to their associated sub-AST node indices. Another example is a mapping between the 
identifiers' indices and the AST leaves (terminals) they appear in. Moreover, the AST paths themselves are stored as 
sequences of AST nodes indices. The same goes for the CFG paths as well.

The [preprocessed code task input tensors](ndfa/code_nn_modules/code_task_input.py#L121) dataclasses well-demonstrate 
this concept. Note how in the class `PDGInputTensors` the field `cfg_nodes_control_kind` mentions the name `cfg_nodes` 
as `self_indexing_group`. That means that the i-th element there corresponds with the relevant cfg-node, and by so 
defines the indexing manner for the cfg nodes. While in the class `CFGPathsInputTensors`, the field `nodes_indices` 
references these indices. When batching several preprocessed methods into a single batch, each can potentially have 
different number of cfg nodes and different number of cfg-paths. Nevertheless, the cfg nodes of all methods will be 
batched in the field `PDGInputTensors.cfg_nodes_control_kind` in a single tensor, and the tensor 
`CFGPathsInputTensors.nodes_indices` will contained the correctly offsetted indices. Additionally, it will contain the
index of the input example (in the batch) for each path.
```python
# Indexing definition:
@dataclasses.dataclass
class PDGInputTensors(TensorsDataClass):
    # (nr_cfg_nodes_in_batch, )
    cfg_nodes_control_kind: Optional[BatchFlattenedTensor] = \
        batch_flattened_tensor_field(default=None, self_indexing_group='cfg_nodes')
# Index referencing:
@dataclasses.dataclass
class CFGPathsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq = \
        batched_flattened_indices_flattened_seq_field(tgt_indexing_group='cfg_nodes')
```

Usually, training and evaluating neural networks is performed over batches of examples, following the SIMD (single 
instruction multiple data) computational scheme to maximize the utility of the accelerated processing units and make 
the training feasible under the available resources. However, the preprocessed example is stored on its own, while it 
should reoccur in various batches during training. We cannot preliminarily perform the batching during preprocessing, 
as the batches should simulate i.i.d. sampling of the entire dataset (and thus each "sampled batch" contains a random 
set of samples, which is determined only during the training and unique per each batch). Therefore, the batching takes 
place during data loading while the training. Whenever a collection of examples is being collated into a batch, 
contiguous flattened tensors are being created containing all the elements in the batch. As a result, the indices of 
these elements are changed. For example, the new index of the 1st element of the 2nd input-sample in the batch (which 
was 0) is now number of items in the 1st sample. Thus, the references to these have to be fixed correspondingly (an 
offset should be added) to retain the indexing semantics. Therefore, batching these (indexed) inputs poses a technical 
challenge. Therefore, we created the [Tensors Data Class](ndfa/misc/tensors_data_class) framework to perform these 
fixes seamlessly and automatically, while requiring zero ad-hoc coding effort.

## Expand-Collapse path-based graph encoding framework
We extended the paths-based graph encoder (originally suggested by Alon et al. in [`code2seq`](https://github.com/tech-srl/code2seq)). Our 
*Expand-Collapse* graph encoding framework expands the input graph into paths, uses sequential encoder to process it 
(propagate information along the paths individually), and then collapses the graph back into nodes representation 
(encodings of node occurrences scattered along paths are folded back into single node representation).

We integrate this approach in the hierarchic model both as a *micro* operator (applied over the top-level expressions 
sub-ASTs) and as a *macro* operator (applied over the CFG).
![Expand Collapse Framework Figure](https://gitfront.io/r/user-5760758/c3d4b342f8f48fe8e3764b2c30925ea140b99535/NDFA/raw/doc/figures/expand-collapse-framework-figure.webp "Expand Collapse Framework Figure")

## Advanced scattered operators
Some of our architectures (mainly the hierarchical model) conceal operations over scattered set of elements (eg: 
AST/CFG nodes of certain types, identifiers). For example, in the 
[expand-collapse](#expand-collapse-path-based-graph-encoding-framework) framework, after performing the `expand` stage 
and encoding of each path separately, each node may participate in multiple paths and in various locations within each 
such containing path. Then, in the `collapse` stage, the encodings of these scattered occurrences should be combined 
into a single node representation. One of our combination methods is attention-based. To support this option, we 
implemented the scattered 
[[classic](ndfa/nn_utils/modules/scatter_attention.py),
[self](ndfa/nn_utils/modules/scatter_self_attention.py),
[general](ndfa/nn_utils/modules/scatter_general_attention.py)]
attention operation.
