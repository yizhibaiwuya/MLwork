import os


def get_classlist(args, phase="train"):
    if args.dataset == 'CIFAR10':
        classes = os.listdir(os.path.join(args.dir, args.dataset, phase))
    
    elif args.dataset == 'CIFAR100':
        csv_path = os.path.join(args.dir, args.dataset, phase + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        inverse_images = {}  # {key: image label (str), value: images path (list)}
        for line in lines:
            img, img_label = line.split(',')
            img_path = os.path.join(args.dir, args.dataset, 'images', img)
            if img_label not in inverse_images.keys():
                inverse_images.update({img_label: []})
            inverse_images.get(img_label).append(img_path)

        classes = list(inverse_images.keys())
    else:
        raise NotImplementedError
    
    return classes


def get_datalist(args, phase, class_list, start_label):
    datalist = []

    if args.dataset == 'CIFAR10':
        classes = os.listdir(os.path.join(args.dir, args.dataset, phase))
        for c in class_list:
            imgs = os.listdir(os.path.join(args.dir, args.dataset, phase, c))
            for img in imgs:
                img_path = os.path.join(args.dir, args.dataset, phase, c, img)
                datalist.append({'file_name': img_path, 'label': start_label})
            start_label += 1

    elif args.dataset == 'CIFAR100':
        csv_path = os.path.join(args.dir, args.dataset, phase + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        # Get all images
        inverse_images = {}  # {key: image label (str), value: images path (list)}
        for line in lines:
            img, img_label = line.split(',')
            img_path = os.path.join(args.dir, args.dataset, 'images', img)
            if img_label not in inverse_images.keys():
                inverse_images.update({img_label: []})
            inverse_images.get(img_label).append(img_path)

        for c in class_list:
            for img_path in inverse_images.get(c):
                datalist.append({'file_name': img_path, 'label': start_label})
            start_label += 1
        classes = list(inverse_images.keys())
    else:
        raise NotImplementedError

    return datalist



if __name__ == "__main__":
    args = {
        'dir': './dataset',  # dataset path: str
        'dataset': 'miniImageNet',  # dataset name: str
    }
    class_list = ['n07613480', 'n04522168', 'n04149813']
    start_label = 3
    phase = 'test'  # train or test: str

    datalist, labels = get_datalist(args=args,
                                    class_list=class_list,
                                    start_label=start_label,
                                    phase=phase
                                    )
    # print(datalist)
    # print(labels)
