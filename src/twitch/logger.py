import torch

printable_encoder = {"geomix_1", "geomix_2", "geomix_3"}


def print_method(args):
    print_str = ""
    if args.method == "augTrain":
        print_str += f"label_grad:{args.label_grad}, aug_lamb:{args.aug_lamb}"
        print_str += "\n"

    return print_str


def print_encoder(args):
    print_str = ""
    if args.encoder in ["geomix_1", "geomix_2"]:
        print_str += f"alpha:{args.alpha} hops:{args.hops}"
    elif args.encoder == "geomix_3":
        print_str += f"res_weight:{args.res_weight}, hops:{args.hops}, graph_wegiht:{args.graph_weight}, use_weight:{args.use_weight}, attn_emb_dim:{args.attn_emb_dim}"

    return print_str


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"Highest In Test: {result[:, 2].max():.2f}")
            for i in range(result.size(1) - 3):
                print(f"Highest OOD Test: {result[:, i+3].max():.2f}")
            print(f"Chosen epoch: {argmax+1}")
            print(f"Final Train: {result[argmax, 0]:.2f}")
            print(f"Final In Test: {result[argmax, 2]:.2f}")
            for i in range(result.size(1) - 3):
                print(f"Final OOD Test: {result[argmax, i+3]:.2f}")
            self.test = result[argmax, 2]
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                train_high = r[:, 0].max().item()
                valid_high = r[:, 1].max().item()
                test_in_high = r[:, 2].max().item()
                test_ood_high = []
                for i in range(r.size(1) - 3):
                    test_ood_high += [r[:, i + 3].max().item()]
                train_final = r[r[:, 1].argmax(), 0].item()
                test_in_final = r[r[:, 1].argmax(), 2].item()
                test_ood_final = []
                for i in range(r.size(1) - 3):
                    test_ood_final += [r[r[:, 1].argmax(), i + 3].item()]
                best_result = (
                    [train_high, valid_high, test_in_high]
                    + test_ood_high
                    + [train_final, test_in_final]
                    + test_ood_final
                )
                best_results.append(best_result)

            best_result = torch.tensor(best_results)
            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"Highest In Test: {r.mean():.2f} ± {r.std():.2f}")
            ood_size = result[0].size(1) - 3
            for i in range(ood_size):
                r = best_result[:, i + 3]
                print(f"Highest OOD Test: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, ood_size + 3]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, ood_size + 4]
            print(f"   Final In  Test: {r.mean():.2f} ± {r.std():.2f}")
            for i in range(ood_size):
                r = best_result[:, i + 5 + ood_size]
                print(f"   Final OOD Test: {r.mean():.2f} ± {r.std():.2f}")

    def output(self, args):
        result = 100 * torch.tensor(self.results)  # [runs, epochs, d], d = 3 + #ood
        best_results = []
        for r in result:
            train_high = r[:, 0].max().item()
            valid_high = r[:, 1].max().item()
            test_in_high = r[:, 2].max().item()
            test_ood_high = []
            for i in range(r.size(1) - 3):
                test_ood_high += [r[:, i + 3].max().item()]
            train_final = r[r[:, 1].argmax(), 0].item()
            test_in_final = r[r[:, 1].argmax(), 2].item()
            test_ood_final = []
            for i in range(r.size(1) - 3):
                test_ood_final += [r[r[:, 1].argmax(), i + 3].item()]
            best_result = (
                [train_high, valid_high, test_in_high]
                + test_ood_high
                + [train_final, test_in_final]
                + test_ood_final
            )
            best_results.append(best_result)
        best_result = torch.tensor(best_results)

        ood_size = result[0].size(1) - 3

        print_str = ""
        r = best_result[:, ood_size + 3]
        print_str += f"Train:{r.mean():.2f} ± {r.std():.2f}, "
        r = best_result[:, ood_size + 4]
        print_str += f"In Test:{r.mean():.2f} ± {r.std():.2f}, "
        print_str += "OOD Test:[ "
        for i in range(ood_size):
            r = best_result[:, i + 5 + ood_size]
            print_str += f"{r.mean():.2f} ± {r.std():.2f}"
            if i < ood_size - 1:
                print_str += ", "
        print_str += " ]"

        print(print_str)
        return print_str
