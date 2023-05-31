class MIRISWorkload:
    def __init__(self, path: str, src_fps: int=10, dst_fps: int=30) -> None:
        self.k = dst_fps / src_fps
        self.workload = []
        visited = set()
        with open(path, 'r') as f:
            lines = f.readlines()
        sub_workload = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                if len(sub_workload) > 0:
                    self.workload.append(sub_workload)
                sub_workload = []
            else:
                t = int(int(line) * self.k)
                if True: #t not in visited:
                    sub_workload.append(t)
                    visited.add(t)
        if len(sub_workload) > 0:
            self.workload.append(sub_workload)

    def __len__(self):
        return len(self.workload)
    
    def __getitem__(self, idx):
        return self.workload[idx]

if __name__ == '__main__':
    workload = MIRISWorkload('query/miris/cache/canal.txt')
    for i, w in enumerate(workload):
        print(i)