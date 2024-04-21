from concurrent.futures import ThreadPoolExecutor
import re
import os
from .prompts import initial_gemm_prompt, compile_error_prompt, runtime_error_prompt, correctness_error_prompt, improve_performance_prompt, initial_my_sgemm, initial_my_sgemm_runner
import asyncio
from transformers import AutoTokenizer
import subprocess


class GemmEnv:
    def __init__(self, tokenizer_name, code_base_dir, max_steps=1, idx=0):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.steps = 0
        self.max_steps = max_steps
        self.messages = []
        self.src_dir = f"{code_base_dir}_{idx:02d}/src"
        self.build_dir = f"{code_base_dir}_{idx:02d}/build"

    def get_named_code_blocks(self, resp):
        named_code_blocks = re.findall(r"`(\S+)`:\n```.*?\n(.+?)\n```", resp, re.DOTALL)
        return named_code_blocks

    def reset_code(self):
        with open(os.path.join(self.src_dir, "my_sgemm.cuh"), "w") as f:
            f.write(initial_my_sgemm)
        with open(os.path.join(self.src_dir, "my_sgemm_runner.cuh"), "w") as f:
            f.write(initial_my_sgemm_runner)

    def update_code(self, blocks):
        for name, code in blocks:
            with open(os.path.join(self.src_dir, name), "w") as f:
                f.write(code)

    def compile_benchmark(self):
        proc = subprocess.Popen(["cmake", "--build", "."], cwd=self.build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()
        compile_error = "error" in stderr.lower() or proc.returncode != 0
        return compile_error, stdout, stderr
        # return False, "", ""

    def run_benchmark(self):
        proc = subprocess.Popen(["./sgemm", "1"], cwd=self.build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()
        runtime_error = "correctness" not in stdout.lower() and proc.returncode != 0
        correctness_error = "correctness" in stdout.lower() and proc.returncode != 0
        return runtime_error, correctness_error, stdout, stderr
        # return False, False, "", ""

    def reset(self):
        self.steps = 0
        self.messages = []
        self.messages.append({
            "role": "user",
            "content": initial_gemm_prompt()
        })
        self.reset_code()
        return self.tokenizer.apply_chat_template(self.messages, tokenize=False)
    
    def _step(self, action):
        self.messages.append({
            "role": "assistant",
            "content": action
        })
        print(">>> Received action:")
        print(action)
        blocks = self.get_named_code_blocks(action)
        self.update_code(blocks)
        compile_error, stdout, stderr = self.compile_benchmark()
        if compile_error:
            print("COMPILE ERROR")
            response = compile_error_prompt(stdout, stderr)
            reward = -10
        else:
            print("NO COMPILE ERROR")
            runtime_error, correctness_error, stdout, stderr = self.run_benchmark()
            if runtime_error:
                print("RUNTIME ERROR")
                response = runtime_error_prompt(stdout, stderr)
                reward = -10
            elif correctness_error:
                print("CORRECTNESS ERROR")
                response = correctness_error_prompt(stdout, stderr)
                reward = -10
            else:
                print("NO CORRECTNESS ERROR")
                response = improve_performance_prompt(stdout, stderr)
                pattern = r"%cuBLAS:\s+([\d.]+)%"
                scores = [float(match) for match in re.findall(pattern, stdout)]
                scores = scores[2:]
                average_score = sum(scores) / len(scores) if scores else 0
                reward = average_score
                print("RECEIVED REWARD = ", reward)

        self.messages.append({
            "role": "user",
            "content": response
        })
        self.steps += 1
        done = self.steps == self.max_steps
        print(">>> New message history:")
        print(self.tokenizer.apply_chat_template(self.messages, tokenize=False))
        return self.tokenizer.apply_chat_template(self.messages, tokenize=False), reward, done


def step_env(env_action_pair):
    env, action = env_action_pair
    return env._step(action)

class BatchedGemmEnv:
    def __init__(self, tokenizer, code_base_dir, max_steps=1, bsize=32):
        self.env_list = [GemmEnv(tokenizer, code_base_dir, max_steps=max_steps, idx=i) for i in range(bsize)]
        self.bsize = bsize

    def reset(self, idx):
        return [env.reset() for env in self.env_list]
    
    def step(self, actions):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(step_env, zip(self.env_list, actions)))
