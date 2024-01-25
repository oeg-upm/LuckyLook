import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.models as module_arch
from parse_config import ConfigParser
from model.bert_base import bert_base
from model.bert_gnn import bert_gnn
from model.bert_gnn_PT import bert_gnn_PT


def main(config):
    logger = config.get_logger('test')

    # Assign batch size
    size = 16

    # setup data_loader instances
    # Prepare the base arguments
    data_loader_args = {
        "dir": './Dataset/Complete/data_pymed_train.csv',
        "data_dir": "./Dataset/Complete/data_pymed_val.csv",
        "batch_size": size,
        "shuffle": False,
        "type": config['data_loader']['args']['type'],
        "model": config['data_loader']['args']['model'],                                                           
        "num_classes": int(config['data_loader']['args']['num_classes']),                                                                                                                      
        "validation_split": 0.0,
        "max_length": int(config['data_loader']['args']['max_length'])
    }

    # Conditionally add 'content' if it exists in the config
    if 'content' in config['data_loader']['args']:
        data_loader_args['content'] = config['data_loader']['args']['content']

    # Create the data loader with unpacked arguments
    data_loader = getattr(module_data, "Papers")(**data_loader_args)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            if isinstance(model, bert_base) or isinstance(model, bert_gnn) or isinstance(model, bert_gnn_PT):
                input_ids = data['input_ids']
                mask = data['attention_mask']
                input_ids, mask, target = input_ids.to(device), mask.to(device), target.to(device)
                output = model(input_ids, mask)
            else:    
                data, target = data.to(device), target.to(device)
                output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = size
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
