
data_dir_train = r'Dataset\train'
data_dir_test = r'Dataset\test'

transform_train = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224), 
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transform_test = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224), 
                                transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

dataset_train = datasets.ImageFolder(data_dir_train, transform=transform_train)
dataset_test = datasets.ImageFolder(data_dir_test, transform=transform_test)