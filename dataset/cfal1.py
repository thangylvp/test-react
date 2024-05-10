import json
import os

class CfaQADataset():
    def __init__(self, cfg) -> None:
        self.questions = {}
        self.map_q = {}
        for root, dirs, files in os.walk(cfg['dataset']['root']):
            for file in files:
                name, ext = os.path.splitext(file)
                self.map_q[name] = root.split('/')[-1]
                with open(os.path.join(root, file), "r") as f:
                    self.questions[name] = json.load(f)
                

if __name__ == "__main__":
    tmp = CfaQADataset({"dataset":{"root":"/scratch/thangnv/cfalevel1"}})

    # for name in tmp.questions:
    #     print(name)
    #     print(tmp.questions[name]['question'])
    #     print(tmp.questions[name]['choices'])