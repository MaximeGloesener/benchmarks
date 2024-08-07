"""
To rank models, it is cool to compare them based on accuracy. 
The easy way to do this is to scrap the accuracy from the model's documentation. 
Link = https://pytorch.org/vision/stable/models.html
"""


import csv

# The list of model accuracies we generated earlier
accuracy_dict = {
    "AlexNet": (56.522, 79.066),
    "ConvNeXt_Base": (84.062, 96.87),
    "ConvNeXt_Large": (84.414, 96.976),
    "ConvNeXt_Small": (83.616, 96.65),
    "ConvNeXt_Tiny": (82.52, 96.146),
    "DenseNet121": (74.434, 91.972),
    "DenseNet161": (77.138, 93.56),
    "DenseNet169": (75.6, 92.806),
    "DenseNet201": (76.896, 93.37),
    "EfficientNet_B0": (77.692, 93.532),
    "EfficientNet_B1": (79.838, 94.934),
    "EfficientNet_B2": (80.608, 95.31),
    "EfficientNet_B3": (82.008, 96.054),
    "EfficientNet_B4": (83.384, 96.594),
    "EfficientNet_B5": (83.444, 96.628),
    "EfficientNet_B6": (84.008, 96.916),
    "EfficientNet_B7": (84.122, 96.908),
    "EfficientNet_V2_L": (85.808, 97.788),
    "EfficientNet_V2_M": (85.112, 97.156),
    "EfficientNet_V2_S": (84.228, 96.878),
    "GoogLeNet": (69.778, 89.53),
    "Inception_V3": (77.294, 93.45),
    "MNASNet0_5": (67.734, 87.49),
    "MNASNet0_75": (71.18, 90.496),
    "MNASNet1_0": (73.456, 91.51),
    "MNASNet1_3": (76.506, 93.522),
    "MaxVit_T": (83.7, 96.722),
    "MobileNet_V2": (72.154, 90.822),
    "MobileNet_V3_Large": (75.274, 92.566),
    "MobileNet_V3_Small": (67.668, 87.402),
    "RegNet_X_16GF": (82.716, 96.196),
    "RegNet_X_1_6GF": (79.668, 94.922),
    "RegNet_X_32GF": (83.014, 96.288),
    "RegNet_X_3_2GF": (81.196, 95.43),
    "RegNet_X_400MF": (74.864, 92.322),
    "RegNet_X_800MF": (77.522, 93.826),
    "RegNet_X_8GF": (81.682, 95.678),
    "RegNet_Y_128GF": (88.228, 98.682),
    "RegNet_Y_16GF": (86.012, 98.054),
    "RegNet_Y_1_6GF": (80.876, 95.444),
    "RegNet_Y_32GF": (86.838, 98.362),
    "RegNet_Y_3_2GF": (81.982, 95.972),
    "RegNet_Y_400MF": (75.804, 92.742),
    "RegNet_Y_800MF": (78.828, 94.502),
    "RegNet_Y_8GF": (82.828, 96.33),
    "ResNeXt101_32X8D": (82.834, 96.228),
    "ResNeXt101_64X4D": (83.246, 96.454),
    "ResNeXt50_32X4D": (81.198, 95.34),
    "ResNet101": (81.886, 95.78),
    "ResNet152": (82.284, 96.002),
    "ResNet18": (69.758, 89.078),
    "ResNet34": (73.314, 91.42),
    "ResNet50": (80.858, 95.434),
    "ShuffleNet_V2_X0_5": (60.552, 81.746),
    "ShuffleNet_V2_X1_0": (69.362, 88.316),
    "ShuffleNet_V2_X1_5": (72.996, 91.086),
    "ShuffleNet_V2_X2_0": (76.23, 93.006),
    "SqueezeNet1_0": (58.092, 80.42),
    "SqueezeNet1_1": (58.178, 80.624),
    "Swin_B": (83.582, 96.64),
    "Swin_S": (83.196, 96.36),
    "Swin_T": (81.474, 95.776),
    "Swin_V2_B": (84.112, 96.864),
    "Swin_V2_S": (83.712, 96.816),
    "Swin_V2_T": (82.072, 96.132),
    "VGG11_BN": (70.37, 89.81),
    "VGG11": (69.02, 88.628),
    "VGG13_BN": (71.586, 90.374),
    "VGG13": (69.928, 89.246),
    "VGG16_BN": (73.36, 91.516),
    "VGG16": (71.592, 90.382),
    "VGG19_BN": (74.218, 91.842),
    "VGG19": (72.376, 90.876),
    "ViT_B_16": (85.304, 97.65),
    "ViT_B_32": (75.912, 92.466),
    "ViT_H_14": (88.552, 98.694),
    "ViT_L_16": (88.064, 98.512),
    "ViT_L_32": (76.972, 93.07),
    "Wide_ResNet101_2": (82.51, 96.02),
    "Wide_ResNet50_2": (81.602, 95.758)
}

# lower case the model names
accuracy_dict = {k.lower(): v for k, v in accuracy_dict.items()}

# Read the input file and write to a new CSV file
with open('GTX3060.txt', 'r') as infile, open('output_file.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=',')
    writer = csv.writer(outfile)
    
    for row in reader:
        name = row[0].lower()
        if name in accuracy_dict:
            acc_top1, acc_top5 = accuracy_dict[name]
            writer.writerow(row + [acc_top1, acc_top5])
        else:
            writer.writerow(row + ['N/A', 'N/A'])
