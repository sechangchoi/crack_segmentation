from Team21_models import *
from Team21_datasets import *
from Team21_datasets import dataLoad as dl
from Team21_trains import * 
import matplotlib.pyplot as plt
from Team21_trains import main as trainMain

# argparse
def define():
    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, default = 'Unet++')
    p.add_argument('--encoder', type=str)
    p.add_argument('--checkPath', type=str)

    modelLoad = p.parse_args()

    return modelLoad

def main(modelLoad):


    # check_path = modelLoad.encoder
    check_path = modelLoad.checkPath


    print(check_path)

    if modelLoad.model == 'Unet':
        model= Team21_Model(arch = 'Unet', encoder_name = modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, arch = modelLoad.model, encoder_name=modelLoad.encoder)
        print(modelLoad.model)
    elif modelLoad.model == 'Unet++':
        model= Team21_Model("UnetPlusPlus", encoder_name = modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, arch = 'unetplusplus', encoder_name=modelLoad.encoder)
        print(modelLoad.model)
    elif modelLoad.model == 'DeepLabV3':
        model= Team21_Model('DeepLabV3', modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, modelLoad.model, encoder_name=modelLoad.encoder)
        print(modelLoad.model)
    elif modelLoad.model == 'DeepLabV3+':
        model= Team21_Model('DeepLabV3plus', modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, 'DeepLabV3plus', encoder_name=modelLoad.encoder)
        print(modelLoad.model)
    elif modelLoad.model == 'FPN':
        model= Team21_Model('FPN', modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, modelLoad.model, encoder_name=modelLoad.encoder)
        print(modelLoad.model)
    elif modelLoad.model == 'PSPNet':
        model= Team21_Model('PSPNet', modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, modelLoad.model, encoder_name=modelLoad.encoder)
        print(modelLoad.model)
    elif modelLoad.model == 'PAN':
        model= Team21_Model('PAN', modelLoad.encoder, in_channels=3, out_classes=1).to(trainMain.device)
        model = model.load_from_checkpoint(check_path, modelLoad.model, encoder_name=modelLoad.encoder)
        print(modelLoad.model)

    print(dl.df.shape)
    train_loader, valid_loader, test_loader = prepare_loaders(df = dl.df, 
                                                    train_num= int(dl.df.shape[0] * .7), 
                                                    test_num= int(dl.df.shape[0] * .86), 
                                                    bs = 1)
    
    # valid_metrics = pl.Trainer.validate(model, dataloaders=valid_loader,  ckpt_path=check_path, verbose=True)
    # pprint(valid_metrics)

    # test_metrics = pl.Trainer.test(model, dataloaders=test_loader, ckpt_path=check_path, verbose=True)
    # pprint(test_metrics)

    batch = next(iter(test_loader))
    with torch.no_grad():
        model.eval()
        logits = model(batch[0])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.detach().cpu().numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.squeeze().detach().cpu().numpy()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.squeeze().numpy()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()
        # test
    


if __name__ == '__main__':
    modelLoad = define()
    main(modelLoad)  
