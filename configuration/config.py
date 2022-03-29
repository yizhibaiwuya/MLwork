import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="[CIFAR10, CIFAR100]",
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="/home/xyk/CIFAR100"
    )
    
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )

    

    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="[Conv64F, resnet12, resnet18]"  
    )

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="[adam, sgd]")
    parser.add_argument("--sched_name", type=str, default="cos", help="[cos, anneal]")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=5, help="Epoch")

    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--initial_annealing_period",
        type=int,
        default=20,
        help="Initial Period that does not anneal",
    )
    parser.add_argument(
        "--annealing_period",
        type=int,
        default=20,
        help="Period (Epochs) of annealing lr",
    )
    parser.add_argument(
        "--learning_anneal", type=float, default=10, help="Divisor for annealing"
    )



    args = parser.parse_args()
    return args
