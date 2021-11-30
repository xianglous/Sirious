def main():
    # WandB – Initialize a new run
    wandb.init(project="transformers_tutorials_summarization")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 2        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1 
    config.LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 1024
    config.SUMMARY_LEN = 150

    free_gpu_cache()
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    # df = pd.read_csv('/content/gdrive/MyDrive/ted-summaries-clean.csv',encoding='latin-1', sep='delimiter')
    # df = pd.read_csv('/content/gdrive/MyDrive/ted-summaries-clean.csv',encoding='latin-1', error_bad_lines=False)
    df = pd.read_csv('/content/gdrive/MyDrive/ted-summaries-1128-1220am.csv',encoding='latin-1')
    print("hello")
    print(list(df.columns))
    df = df[["summary","transcript"]]
    df.transcript = 'summarize: ' + df.transcript
    print(df.head())

    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state = config.SEED)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)


    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('/content/gdrive/MyDrive/predictions_3.csv')
        print('Output Files generated for review')

if __name__ == '__main__':
    main()



'''
wandb: Currently logged in as: xytang (use `wandb login --relogin` to force relogin)
Syncing run frosty-flower-58 to Weights & Biases (docs).
Initial GPU Usage
| ID | GPU | MEM |
------------------
|  0 |  0% |  0% |
GPU Usage after emptying the cache
| ID | GPU | MEM |
------------------
|  0 |  5% |  1% |
hello
['title', 'url_summary', 'url_ted', 'month', 'speaker', 'year', 'summary', 'transcript']
                                             summary                                         transcript
0  When we imagine corruption it tends to be a mi...  summarize: When we talk about corruption, ther...
1  Often it is said people are too stupid, selfis...  summarize: How often do we hear that people ju...
2  Ken Jennings loved game shows from a young age...  summarize: In two weeks time, that's the ninth...
3  Marco introduces artificial reality glasses th...  summarize: Good morning. So magic is an excell...
4  Sustainability is the investment logic looking...  summarize: The world is changing in some reall...
FULL Dataset: (131, 2)
TRAIN Dataset: (105, 2)
TEST Dataset: (26, 2)
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Initiating Fine-Tuning for the model on our dataset
/usr/local/lib/python3.7/dist-packages/transformers/tokenization_utils_base.py:2218: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
  FutureWarning,
Epoch: 0, Loss:  10.451733589172363
Epoch: 1, Loss:  3.246710777282715
Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe
Completed 0
Output Files generated for review
'''


'''
Finishing last run (ID:6omdku8u) before initializing another...

Waiting for W&B process to finish, PID 1747... (success).
Run history:

Training Loss	█▃▂▂▂▂▁▂▁▁▁▁

Run summary:

Training Loss	2.90551
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Synced frosty-flower-58: https://wandb.ai/xytang/transformers_tutorials_summarization/runs/6omdku8u
Find logs at: ./wandb/run-20211128_065328-6omdku8u/logs
Successfully finished last run (ID:6omdku8u). Initializing new run:
Syncing run charmed-pond-59 to Weights & Biases (docs).
Initial GPU Usage
| ID | GPU | MEM |
------------------
|  0 |  0% | 99% |
GPU Usage after emptying the cache
| ID | GPU | MEM |
------------------
|  0 |  6% |  1% |
hello
['title', 'url_summary', 'url_ted', 'month', 'speaker', 'year', 'summary', 'transcript']
                                             summary                                         transcript
0  When we imagine corruption it tends to be a mi...  summarize: When we talk about corruption, ther...
1  Often it is said people are too stupid, selfis...  summarize: How often do we hear that people ju...
2  Ken Jennings loved game shows from a young age...  summarize: In two weeks time, that's the ninth...
3  Marco introduces artificial reality glasses th...  summarize: Good morning. So magic is an excell...
4  Sustainability is the investment logic looking...  summarize: The world is changing in some reall...
FULL Dataset: (131, 2)
TRAIN Dataset: (105, 2)
TEST Dataset: (26, 2)
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Initiating Fine-Tuning for the model on our dataset
/usr/local/lib/python3.7/dist-packages/transformers/tokenization_utils_base.py:2218: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
  FutureWarning,
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-27-9c99b7491f94> in <module>()
    100 
    101 if __name__ == '__main__':
--> 102     main()

12 frames
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py in softmax(input, dim, _stacklevel, dtype)
   1678         dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
   1679     if dtype is None:
-> 1680         ret = input.softmax(dim)
   1681     else:
   1682         ret = input.softmax(dim, dtype=dtype)

RuntimeError: CUDA out of memory. Tried to allocate 384.00 MiB (GPU 0; 11.17 GiB total capacity; 10.41 GiB already allocated; 80.81 MiB free; 10.51 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
'''



'''
You have basically three options:
You cut the longer texts off and only use the first 512 Tokens. The original BERT implementation (and probably the others as well) truncates longer sequences automatically. For most cases, this option is sufficient.
You can split your text in multiple subtexts, classifier each of them and combine the results back together ( choose the class which was predicted for most of the subtexts for example). This option is obviously more expensive.
You can even feed the output token for each subtext (as in option 2) to another network (but you won't be able to fine-tune) as described in this discussion.
I would suggest to try option 1, and only if this is not good enough to consider the other options.
'''