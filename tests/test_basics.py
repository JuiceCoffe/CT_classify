"""离线测试用合成张量验证逻辑，不把测试夹具当成真实实验成绩。"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import uuid
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nodule_lab.data import NoduleDataset
from nodule_lab.augment import RandomFlip, GaussianNoise, RandomContrast, Compose
from nodule_lab.model import TinyCNN
from nodule_lab.metrics import classification_metrics
from nodule_lab.engine import train, evaluate, load_model, seed_everything, resolve_device
from prepare_data import check_md5
from train import read_config


class Fixture(unittest.TestCase):
    """构造分布与官方数据无关的小型离线夹具，用于程序正确性检查。"""
    def setUp(self):
        temp_root = ROOT/'outputs'/'test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        # 普通mkdir继承目录权限，兼容Windows受限账户；每次仅清理自己的夹具目录。
        self.folder = temp_root/('case_'+uuid.uuid4().hex)
        self.folder.mkdir()
        assert self.folder.resolve().parent == temp_root.resolve()
        self.addCleanup(shutil.rmtree, self.folder)
        self.path = self.folder/'fixture.npz'
        rng = np.random.default_rng(7)
        self.arrays = {}
        for split, n in [('train',10),('val',6),('test',8)]:
            self.arrays[split+'_images'] = rng.integers(0,256,(n,28,28,28),dtype=np.uint8)
            self.arrays[split+'_labels'] = (np.arange(n)%2).astype(np.uint8)[:,None]
        np.savez(self.path, **self.arrays)
        self.config = json.loads((ROOT/'configs/baseline.json').read_text())
        self.config.update(data=str(self.path), output=str(self.folder/'run'), epochs=2,
                           batch_size=4, device='cpu', threads=2)
        seed_everything(42, 2)


class DataTests(Fixture):
    def test_center_slice_and_normalize_once(self):
        dataset = NoduleDataset(self.path,'train')
        image,label = dataset[0]
        self.assertEqual(image.shape,(1,28,28))
        self.assertEqual(image.dtype,torch.float32)
        self.assertEqual(label.dtype,torch.float32)
        np.testing.assert_array_equal(image.numpy()[0],self.arrays['train_images'][0,14].astype(np.float32)/255)

    def test_official_split_names_and_lengths(self):
        for name,n in [('train',10),('val',6),('test',8)]:
            self.assertEqual(len(NoduleDataset(self.path,name)),n)
        with self.assertRaisesRegex(ValueError,'split'):
            NoduleDataset(self.path,'validation')

    def test_reject_val_and_test_augmentation(self):
        for split in ['val','test']:
            with self.assertRaisesRegex(ValueError,'train-only'):
                NoduleDataset(self.path,split,RandomFlip())

    def test_transform_does_not_pollute_source_or_label(self):
        dataset = NoduleDataset(self.path,'train',lambda x:x.zero_())
        original=dataset.images.clone()
        image,label=dataset[1]
        self.assertEqual(image.sum().item(),0)
        self.assertEqual(label.item(),1)
        torch.testing.assert_close(dataset.images,original)

    def test_val_is_repeatable(self):
        dataset=NoduleDataset(self.path,'val')
        torch.testing.assert_close(dataset[2][0],dataset[2][0],rtol=0,atol=0)

    def test_invalid_dtype_shape_label_and_count(self):
        invalid=[('train_images',np.zeros((10,28,28,28),dtype=np.float32)),
                 ('train_images',np.zeros((10,27,28,28),dtype=np.uint8)),
                 ('train_labels',np.ones((10,1),dtype=np.uint8)*2),
                 ('train_labels',np.zeros((9,1),dtype=np.uint8))]
        for key,value in invalid:
            with self.subTest(key=key,shape=value.shape):
                arrays=dict(self.arrays);arrays[key]=value
                np.savez(self.path,**arrays)
                with self.assertRaises(ValueError):NoduleDataset(self.path,'train')

    def test_transform_contract_shape_and_range(self):
        for transform in [lambda x:x.squeeze(0),lambda x:x+2,lambda x:x*float('nan')]:
            with self.assertRaises(ValueError):NoduleDataset(self.path,'train',transform)[0]

    def test_md5_rejects_synthetic_fixture(self):
        with self.assertRaisesRegex(ValueError,'MD5 expected'):
            check_md5(self.path)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image=torch.linspace(0,1,28*28).reshape(1,28,28)

    def test_disabled_transforms_are_identity(self):
        for transform in [RandomFlip(0),GaussianNoise(0),RandomContrast(0)]:
            torch.testing.assert_close(transform(self.image),self.image,rtol=0,atol=0)

    def test_flip_both_axes_exactly(self):
        torch.testing.assert_close(RandomFlip(1)(self.image),self.image.flip(-1,-2))

    def test_noise_changes_input_and_clamps(self):
        original=self.image.clone();torch.manual_seed(9)
        result=GaussianNoise(1,(.5,.5))(self.image)
        self.assertFalse(torch.equal(result,self.image))
        self.assertTrue(0<=result.min()<=result.max()<=1)
        torch.testing.assert_close(original,self.image)

    def test_zero_noise_is_identity(self):
        torch.testing.assert_close(GaussianNoise(1,(0,0))(self.image),self.image)

    def test_contrast_formula(self):
        actual=RandomContrast(1,(.5,.5))(self.image)
        expected=self.image.mean()+.5*(self.image-self.image.mean())
        torch.testing.assert_close(actual,expected)

    def test_contrast_constant_image(self):
        image=torch.full((1,28,28),.4)
        torch.testing.assert_close(RandomContrast(1,(0,2))(image),image)

    def test_compose_seed_reproducibility(self):
        transform=Compose([{'name':'flip'},{'name':'noise','p':1},{'name':'contrast','p':1}])
        torch.manual_seed(17);a=transform(self.image)
        torch.manual_seed(17);b=transform(self.image)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(a.shape,self.image.shape)
        self.assertEqual(a.dtype,torch.float32)

    def test_empty_compose(self):
        torch.testing.assert_close(Compose([])(self.image),self.image)

    def test_bad_config_fails_loudly(self):
        for constructor,args in [(RandomFlip,{'p':2}),(GaussianNoise,{'sigma':(.2,.1)}),
                                 (RandomContrast,{'factor':(-1,1)})]:
            with self.assertRaises(ValueError):constructor(**args)
        with self.assertRaisesRegex(ValueError,'Unknown augmentation'):Compose([{'name':'typo'}])
        with self.assertRaises(TypeError):Compose([{'name':'noise','sigam':[0,1]}])


class MetricsModelTests(unittest.TestCase):
    def test_hand_computed_confusion_and_boundary(self):
        result=classification_metrics([0,0,1,1],[.1,.5,.4,.9])
        self.assertEqual([result[k] for k in ['TP','FP','TN','FN']],[1,1,1,1])
        self.assertEqual(result['AUC'],.75)
        for key in ['ACC','Sensitivity','Specificity','F1','BACC','Precision']:
            self.assertEqual(result[key],.5)

    def test_perfect_prediction(self):
        r=classification_metrics([0,1],[0,1])
        self.assertEqual(r['AUC'],1);self.assertEqual(r['F1'],1)

    def test_zero_division_and_single_class(self):
        r=classification_metrics([0,0],[0,0])
        self.assertIsNone(r['AUC']);self.assertEqual(r['F1'],0)

    def test_invalid_metric_input(self):
        for labels,probs in [([],[]),([0],[.1,.2]),([2],[.5]),([0],[float('nan')]),([1],[2])]:
            with self.assertRaises(ValueError):classification_metrics(labels,probs)

    def test_model_shape_and_parameter_count(self):
        model=TinyCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()),23585)
        for n in [1,4]:self.assertEqual(model(torch.zeros(n,1,28,28)).shape,(n,))
        with self.assertRaisesRegex(ValueError,'TinyCNN image'):model(torch.zeros(1,3,28,28))

    def test_gradients_and_optimizer_update(self):
        torch.manual_seed(42); model=TinyCNN()
        old=model.classifier.weight.detach().clone()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.rand(4,1,28,28)),torch.tensor([0.,1.,0.,1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.features[0].weight.grad).all())
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(),0)
        opt.step();self.assertFalse(torch.equal(old,model.classifier.weight))


class PipelineTests(Fixture):
    def test_evaluate_no_parameter_or_bn_updates_and_repeatable(self):
        model=TinyCNN();before={k:v.clone() for k,v in model.state_dict().items()}
        loader=DataLoader(NoduleDataset(self.path,'val'),batch_size=4)
        _,_,y,a=evaluate(model,loader,torch.device('cpu'),False)
        _,_,_,b=evaluate(model,loader,torch.device('cpu'),False)
        np.testing.assert_array_equal(y,self.arrays['val_labels'][:,0])
        np.testing.assert_array_equal(a,b)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_train_never_loads_test_and_checkpoint_replays_val(self):
        with patch('nodule_lab.engine.NoduleDataset',wraps=NoduleDataset) as reader:
            checkpoint=train(self.config,False)
        self.assertEqual([c.args[1] for c in reader.call_args_list],['train','val'])
        model,payload=load_model(checkpoint,torch.device('cpu'))
        loader=DataLoader(NoduleDataset(self.path,'val'),batch_size=4)
        _,metrics,_,_=evaluate(model,loader,torch.device('cpu'),False)
        self.assertAlmostEqual(metrics['AUC'],payload['val_auc'],places=10)
        with (checkpoint.parent/'history.csv').open() as file:
            history=list(csv.DictReader(file))
        first_best=max(history,key=lambda r:float(r['val_auc']))
        self.assertEqual(payload['epoch'],int(first_best['epoch']))

    def test_early_stop_ties_keep_first_epoch(self):
        self.config.update(epochs=10,patience=2)
        with patch('nodule_lab.engine.evaluate',return_value=(.7,{'AUC':.6,'F1':.5},None,None)):
            checkpoint=train(self.config,False)
        _,payload=load_model(checkpoint,torch.device('cpu'))
        self.assertEqual(payload['epoch'],1)
        done=json.loads((checkpoint.parent/'training_complete.json').read_text())
        self.assertEqual(done['completed_epochs'],3)

    def test_seeded_training_is_repeatable(self):
        self.config['epochs']=1
        a=train(self.config,False)
        self.config['output']=str(self.folder/'second')
        b=train(self.config,False)
        am,_=load_model(a,torch.device('cpu'));bm,_=load_model(b,torch.device('cpu'))
        for key,value in am.state_dict().items():
            torch.testing.assert_close(value,bm.state_dict()[key],rtol=0,atol=0)

    def test_three_augmentations_train_end_to_end(self):
        self.config.update(epochs=1,augmentation=[{'name':'flip'},{'name':'noise'},{'name':'contrast'}])
        self.assertTrue(train(self.config,False).is_file())

    def test_existing_output_not_overwritten(self):
        out=Path(self.config['output']);out.mkdir();(out/'keep.txt').write_text('original')
        with self.assertRaises(FileExistsError):train(self.config,False)
        self.assertEqual((out/'keep.txt').read_text(),'original')

    def test_invalid_training_budget(self):
        self.config['epochs']=0
        with self.assertRaisesRegex(ValueError,'epochs'):train(self.config,False)

    def test_validation_single_class_rejected(self):
        self.arrays['val_labels'][:]=0;np.savez(self.path,**self.arrays)
        with self.assertRaisesRegex(ValueError,'both label classes'):train(self.config,False)

    def test_cli_overrides_config_and_rejects_unknown_keys(self):
        args=argparse.Namespace(config=ROOT/'configs/baseline.json',data=str(self.path),
            output=str(self.folder/'cli'),device='cpu',epochs=1,seed=43)
        cfg=read_config(args)
        self.assertEqual(cfg['epochs'],1);self.assertEqual(cfg['seed'],43)
        self.assertEqual(cfg['device'],'cpu')
        path=self.folder/'bad.json';bad=dict(cfg,epohs=2);path.write_text(json.dumps(bad))
        args.config=path
        with self.assertRaisesRegex(ValueError,'unknown'):read_config(args)

    def test_evaluate_cli_exports_all_samples_and_refuses_overwrite(self):
        self.config['epochs']=1;checkpoint=train(self.config,False)
        output=self.folder/'evaluation'
        command=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',str(checkpoint),
                 '--data',str(self.path),'--output',str(output),'--device','cpu','--split','test']
        cli_env=dict(os.environ, PYTHONIOENCODING='utf-8')
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=cli_env)
        self.assertEqual(result.returncode,0,result.stderr)
        with (output/'predictions.csv').open() as file:predictions=list(csv.DictReader(file))
        self.assertEqual(len(predictions),8)
        self.assertEqual([int(r['index']) for r in predictions],list(range(8)))
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=cli_env)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Output exists',result.stderr)

    def test_device_selection(self):
        self.assertEqual(str(resolve_device('cpu')),'cpu')
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(str(resolve_device('auto')),'cpu')
            with self.assertRaises(RuntimeError):resolve_device('cuda')


@unittest.skipUnless(os.environ.get('NODULE_DATA'),'Set NODULE_DATA for official-data integration tests')
class OfficialDataTests(unittest.TestCase):
    def test_official_md5_and_counts(self):
        path=os.environ['NODULE_DATA'];check_md5(path)
        for split,counts in [('train',[863,295]),('val',[123,42]),('test',[246,64])]:
            dataset=NoduleDataset(path,split)
            actual=np.bincount(dataset.labels.numpy().astype(int),minlength=2).tolist()
            self.assertEqual(actual,counts)

    def test_official_center_slice_exact(self):
        path=os.environ['NODULE_DATA']
        with np.load(path) as z:expected=z['train_images'][0,14].astype(np.float32)/255
        actual=NoduleDataset(path,'train')[0][0].numpy()[0]
        np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':
    unittest.main()
