#!/usr/bin/env python3
"""
Python script to measure memory bandwidth of test_qps_thread using Intel VTune
"""

import os
import subprocess
import json
import re
import sys
from pathlib import Path

def run_command(cmd, cwd = os.getcwd(), filter_vtune_progress=False, verbose=False):
    """Run a command and return the result"""
    if verbose:
        print(f"Running command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd,
                              capture_output=True, text=True, check=True)
        if verbose:
            print(f"Command finished with return code: {result.returncode}")

        # Filter VTune progress messages if requested
        stdout_to_show = result.stdout
        stderr_to_show = result.stderr

        if filter_vtune_progress:
            # Filter out VTune progress lines from stderr
            if stderr_to_show:
                stderr_lines = stderr_to_show.split('\n')
                filtered_stderr = []
                for line in stderr_lines:
                    # Skip VTune progress lines
                    if not (line.startswith('vtune: Executing actions') or
                           line.startswith('vtune: Processing') or
                           'Executing actions' in line and '%' in line):
                        filtered_stderr.append(line)
                stderr_to_show = '\n'.join(filtered_stderr).strip()

        # Only show output if verbose or if there are errors
        if verbose:
            if len(stdout_to_show) > 2000:
                print(f"stdout: [Output truncated - {len(stdout_to_show)} chars total]")
                print(stdout_to_show[:1000] + "\n... [truncated] ...\n" + stdout_to_show[-500:])
            else:
                print(f"stdout: {stdout_to_show}")

            if stderr_to_show:
                if len(stderr_to_show) > 1000:
                    print(f"stderr: [Error output truncated - {len(stderr_to_show)} chars total]")
                    print(stderr_to_show[:500] + "\n... [truncated] ...")
                else:
                    print(f"stderr: {stderr_to_show}")
        elif stderr_to_show:
            # Always show errors even if not verbose
            print(f"Warning - stderr: {stderr_to_show[:200]}{'...' if len(stderr_to_show) > 200 else ''}")

        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout[:500]}{'...' if len(e.stdout) > 500 else ''}")
        if e.stderr:
            print(f"stderr: {e.stderr[:500]}{'...' if len(e.stderr) > 500 else ''}")
        return None, e.stderr

def parse_vtune_memory_bandwidth(report_output):
    """Parse VTune report output to extract memory bandwidth"""
    bandwidth_info = {}

    # Look for the Bandwidth Domain table
    lines = report_output.split('\n')
    in_bandwidth_table = False
    header_found = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Find the bandwidth domain table header
        if 'Bandwidth Domain' in line and 'Platform Maximum' in line:
            header_found = True
            # Skip the separator line (next line with dashes)
            continue

        # Skip the separator line with dashes
        if header_found and line.startswith('----'):
            in_bandwidth_table = True
            continue

        # Parse the bandwidth data lines
        if in_bandwidth_table and line:
            # Stop if we hit an empty line or another section
            if not line or line.startswith('=') or 'Collection and Platform Info' in line:
                break

            # Parse the line - format example:
            # DRAM, GB/sec                      216                         41.200   10.222                                           0.0%
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Extract domain name (first part before comma or space)
                    domain_parts = []
                    unit = ""

                    # Find where the numeric data starts
                    numeric_start = 0
                    for j, part in enumerate(parts):
                        try:
                            float(part.replace('%', ''))
                            numeric_start = j
                            break
                        except ValueError:
                            continue

                    # Everything before numeric data is the domain name and unit
                    domain_text = ' '.join(parts[:numeric_start])

                    # Extract unit if present (e.g., "GB/sec", "%")
                    if ',' in domain_text:
                        domain_name, unit = domain_text.split(',', 1)
                        domain_name = domain_name.strip()
                        unit = unit.strip()
                    else:
                        domain_name = domain_text.strip()
                        unit = ""

                    # Extract numeric values
                    numeric_parts = parts[numeric_start:]
                    if len(numeric_parts) >= 4:
                        platform_max = float(numeric_parts[0])
                        observed_max = float(numeric_parts[1])
                        average = float(numeric_parts[2])
                        high_bw_util_pct = float(numeric_parts[3].replace('%', ''))

                        bandwidth_info[domain_name] = {
                            'unit': unit,
                            'platform_maximum': platform_max,
                            'observed_maximum': observed_max,
                            'average': average,
                            'high_bw_utilization_percent': high_bw_util_pct
                        }

                except (ValueError, IndexError) as e:
                    # Skip lines that can't be parsed
                    continue

    # Also try to extract the legacy bandwidth value for backward compatibility
    legacy_bandwidth = None
    bandwidth_pattern = r"Memory Bound.*?(\d+\.?\d*)\s*GB/s"
    match = re.search(bandwidth_pattern, report_output, re.IGNORECASE)
    if match:
        legacy_bandwidth = float(match.group(1))

    # Alternative patterns for legacy support
    if not legacy_bandwidth:
        traffic_patterns = [
            r"Memory traffic.*?(\d+\.?\d*)\s*GB/s",
            r"DRAM.*?(\d+\.?\d*)\s*GB/s",
            r"Memory bandwidth.*?(\d+\.?\d*)\s*GB/s"
        ]

        for pattern in traffic_patterns:
            match = re.search(pattern, report_output, re.IGNORECASE)
            if match:
                legacy_bandwidth = float(match.group(1))
                break

    return {
        'bandwidth_domains': bandwidth_info,
        'legacy_bandwidth_gbps': legacy_bandwidth
    }

def save_results_and_logs(results, all_logs, results_file, log_file, datasets, B_values, num_threads_values, group, base_args, is_partial=False):
    """Save results and logs to files"""
    # Save results to JSON file
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save all logs to file
    with open(log_file, "w") as f:
        prefix = "=== VTune Memory Bandwidth and QPS Measurement"
        if is_partial:
            prefix += " (Partial Results)"
        else:
            prefix += " Full Log"
        prefix += " ===\n"

        f.write(prefix)
        f.write(f"Date: {os.popen('date').read().strip()}\n")
        f.write(f"Datasets: {datasets}\n")
        f.write(f"B values: {B_values}\n")
        f.write(f"Num threads values: {num_threads_values}\n")
        f.write(f"Group: {group}\n")
        f.write(f"Base args: {base_args}\n")
        f.write("=" * 60 + "\n\n")
        f.writelines(all_logs)

        f.write("\n\n=== CURRENT RESULTS ===\n")
        f.write("Results by B value and thread count:\n\n")
        for dataset in results:
            f.write(f"Dataset: {dataset}\n")
            f.write("-" * 50 + "\n")
            if results[dataset]:
                for B in sorted(results[dataset].keys()):
                    f.write(f"  B = {B}:\n")
                    if results[dataset][B]:
                        for num_threads in sorted(results[dataset][B].keys()):
                            data = results[dataset][B][num_threads]
                            f.write(f"    Threads = {num_threads:2d}:")
                            if 'memory_bandwidth_gbps' in data:
                                f.write(f" BW = {data['memory_bandwidth_gbps']:8.2f} GB/s")
                            if 'qps' in data:
                                f.write(f" QPS = {data['qps']:8.2f}")
                            if 'recall' in data:
                                f.write(f" Recall = {data['recall']:.3f}")
                            f.write("\n")
                    else:
                        f.write("    No successful measurements\n")
            else:
                f.write("No successful measurements\n")
            f.write("\n")

def parse_qps_output(output):
    """Parse test_qps_thread output to extract QPS value"""
    # Look for lines with format: num_threads,QPS,avg_tm_ms,recall,ratio,bw_mbps
    # Example: 4,1145.3312,3.4927382,0.8832,1.0013291,335234.8

    lines = output.split('\n')
    qps_data = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header line
        if 'num_threads,QPS' in line or 'num_threads' in line and 'QPS' in line:
            continue

        # Try to parse CSV format
        parts = line.split(',')
        if len(parts) >= 6:
            try:
                nprobe = int(parts[0])  # First part is num_threads
                num_threads = int(parts[1])
                qps = float(parts[2])
                avg_tm_ms = float(parts[3])
                recall = float(parts[4])
                ratio = float(parts[5])
                bw_mbps = float(parts[6])

                qps_data[num_threads] = {
                    'qps': qps,
                    'avg_tm_ms': avg_tm_ms,
                    'recall': recall,
                    'ratio': ratio,
                    'bw_mbps': bw_mbps
                }
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 4:
            try:
                nprobe = int(parts[0])  # First part is num_threads
                num_threads = int(parts[1])
                qps = float(parts[2])
                recall = float(parts[3])

                qps_data[num_threads] = {
                    'qps': qps,
                    'recall': recall,
                }
            except (ValueError, IndexError):
                continue

    return qps_data

def main():
    # Configuration
    datasets = ["gist"]  # Following the original script
    B_values = [1, 2, 4, 8]  # Extended B values for comprehensive testing
    num_threads_values = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48]  # Thread count values
    group = 4096
    base_args = ""  # Base arguments without num_threads
    base_cmd = "numactl -N 0 -m 0 /home/howarli/dev/SACQ/bin/test_qps_thread"

    datasets = ['msmarco10M']
    B_values = [4, 8, 2, 1]  # Extended B values for comprehensive testing
    base_args = ""  # SAQ
    base_args = "-enable_segmentation=false"  # CAQ

    # B_values = [32]
    # base_cmd = "numactl -N 0 -m 0 /home/howarli/dev/SACQ/bin/test_ivf"

    # datasets = ['laion100m']
    base_args = ""
    # base_cmd = "numactl -N 0 -m 0 /home/howarli/dev/RaBitQ-Library/bin/hnsw_rabitq_querying /home/howarli/dev/SACQ/data/gist/hnsw16_b3_rbq.index /home/howarli/dev/SACQ/data/gist/gist_query.fvecs /home/howarli/dev/SACQ/data/gist/gist_groundtruth.ivecs l2"
    # base_cmd = "numactl -N 0 -m 0 /home/howarli/dev/RaBitQ-Library/bin/hnsw_rabitq_querying /home/howarli/dev/SACQ/data/gist/hnsw16_b3_rbq.index /data/share/users/pqyin/data/laion400m/laion_text_sample.fbin /data/share/users/pqyin/data/laion400m/gt_1000_100M.bin l2"
    base_cmd = "numactl -N 0 -l /home/howarli/dev/RaBitQ-Library/bin/symqg_querying /home/howarli/dev/SACQ/data/gist/sympg_rbq.index /home/howarli/dev/SACQ/data/gist/gist_query.fvecs /home/howarli/dev/SACQ/data/gist/gist_groundtruth.ivecs"
    B_values = [1]

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    bin_dir = project_root / "bin"
    vtune_results_dir = bin_dir / "vtune_results"

    # Ensure vtune results directory exists
    vtune_results_dir.mkdir(exist_ok=True)

    # Create log file for full output
    log_file = vtune_results_dir / "vtune_measurement_log.txt"
    all_logs = []

    # Change to bin directory to run executables
    os.chdir(bin_dir)

    # Results storage
    results = {}

    # Check and source VTune environment
    vtune_env_script = "/opt/intel/oneapi/vtune/latest/env/vars.sh"
    if os.path.exists(vtune_env_script):
        print(f"Sourcing VTune environment from: {vtune_env_script}")
        command = f"source {vtune_env_script} && env"
        process = subprocess.Popen(command, shell=True, executable="/bin/bash",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error sourcing VTune environment: {stderr.decode()}")
            sys.exit(1)

        # Update the environment variables
        for line in stdout.decode().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
        print("VTune environment sourced successfully.")
    else:
        print("VTune environment script not found. Please install Intel oneAPI VTune.")
        sys.exit(1)

    print("=== VTune Memory Bandwidth Measurement ===")
    print(f"Datasets: {datasets}")
    print(f"B values: {B_values}")
    print(f"Num threads values: {num_threads_values}")
    print(f"Group: {group}")
    print(f"Base args: {base_args}")
    print()

    # Define file paths for results
    results_file = vtune_results_dir / "memory_bandwidth_results.json"

    for dataset in datasets:
        results[dataset] = {}

        for B in B_values:
            print(f"\n{'='*20} Starting B = {B} {'='*20}")
            results[dataset][B] = {}

            for num_threads in num_threads_values:
                args = f"{base_args} -num_threads={num_threads}"
                print(f"==========> Measuring for dataset={dataset}, B={B}, num_threads={num_threads}")
                log_entry = f"\n=== Measurement: dataset={dataset}, B={B}, num_threads={num_threads}, args={args} ===\n"
                all_logs.append(log_entry)

                # Construct the test command
                if "hnsw_rabitq_querying" in base_cmd:
                    t = base_cmd.replace('hnsw16_b3_rbq.index', f'hnsw16_b{B}_rbq.index')
                    t = t.replace('gist', dataset)
                    test_cmd = f"{t}  {args}"
                elif "symqg_querying" in base_cmd:
                    t = base_cmd.replace('gist', dataset)
                    test_cmd = f"{t}  {args}"
                else:
                    test_cmd = f"{base_cmd} -dataset {dataset} -B {B} {args}"

                # VTune result directory for this run
                result_name = f"{dataset}_B{B}_T{num_threads}"
                vtune_result_dir = vtune_results_dir / result_name

                # Remove previous results if they exist
                if vtune_result_dir.exists():
                    subprocess.run(f"rm -rf {vtune_result_dir}", shell=True)

                # Run VTune with memory access analysis (filter progress messages)
                vtune_cmd = f"vtune -collect memory-access -result-dir {vtune_result_dir} -- {test_cmd}"

                all_logs.append(f"Command: {vtune_cmd}\n")
                print("  Running VTune collection...")
                stdout, stderr = run_command(vtune_cmd, filter_vtune_progress=True, verbose=False)

                # Log the full output (but don't print to console)
                all_logs.append(f"VTune stdout:\n{stdout}\n")
                if stderr:
                    all_logs.append(f"VTune stderr:\n{stderr}\n")

                if stdout is None:
                    print(f"  ❌ Failed to run VTune for B={B}, num_threads={num_threads}")
                    all_logs.append(f"FAILED: VTune execution for B={B}, num_threads={num_threads}\n")
                    continue

                # Generate VTune report
                report_cmd = f"vtune -report summary -result-dir {vtune_result_dir}"
                all_logs.append(f"Report command: {report_cmd}\n")
                print("  Generating VTune report...")
                report_stdout, report_stderr = run_command(report_cmd, verbose=False)

                # Log the report output
                all_logs.append(f"Report stdout:\n{report_stdout}\n")
                if report_stderr:
                    all_logs.append(f"Report stderr:\n{report_stderr}\n")

                if report_stdout is None:
                    print(f"  ❌ Failed to generate VTune report for B={B}, num_threads={num_threads}")
                    all_logs.append(f"FAILED: VTune report generation for B={B}, num_threads={num_threads}\n")
                    continue

                # Parse memory bandwidth from report
                print("  Parsing VTune report...")
                bandwidth_data = parse_vtune_memory_bandwidth(report_stdout)

                # Parse QPS data from test output
                qps_data = parse_qps_output(stdout)

                if bandwidth_data or qps_data:
                    results[dataset][B][num_threads] = {}
                    success_messages = []

                    if bandwidth_data:
                        # Store the detailed bandwidth information
                        results[dataset][B][num_threads]['bandwidth_analysis'] = bandwidth_data

                        # Extract DRAM bandwidth for summary display
                        dram_bandwidth = None
                        if 'bandwidth_domains' in bandwidth_data:
                            for domain_name, domain_data in bandwidth_data['bandwidth_domains'].items():
                                if 'DRAM' in domain_name.upper():
                                    dram_bandwidth = domain_data['observed_maximum']
                                    break

                        # Fall back to legacy bandwidth if DRAM not found
                        if dram_bandwidth is None and bandwidth_data.get('legacy_bandwidth_gbps'):
                            dram_bandwidth = bandwidth_data['legacy_bandwidth_gbps']

                        if dram_bandwidth is not None:
                            results[dataset][B][num_threads]['memory_bandwidth_gbps'] = dram_bandwidth
                            success_messages.append(f"BW: {dram_bandwidth:.2f} GB/s")
                            all_logs.append(f"SUCCESS: Memory bandwidth for B={B}, T={num_threads}: {dram_bandwidth:.2f} GB/s\n")

                        # Log detailed bandwidth domain information
                        if 'bandwidth_domains' in bandwidth_data and bandwidth_data['bandwidth_domains']:
                            all_logs.append("Detailed bandwidth domain analysis:\n")
                            for domain_name, domain_data in bandwidth_data['bandwidth_domains'].items():
                                all_logs.append(f"  {domain_name} ({domain_data['unit']}): "
                                               f"Platform Max={domain_data['platform_maximum']}, "
                                               f"Observed Max={domain_data['observed_maximum']}, "
                                               f"Average={domain_data['average']}, "
                                               f"High BW Util={domain_data['high_bw_utilization_percent']}%\n")

                    if qps_data:
                        # Find QPS data for the current num_threads
                        if num_threads in qps_data:
                            qps_info = qps_data[num_threads]
                            results[dataset][B][num_threads].update(qps_info)
                            success_messages.append(f"QPS: {qps_info['qps']:.2f}")
                            all_logs.append(f"SUCCESS: QPS for B={B}, T={num_threads}: {qps_info['qps']:.2f}\n")
                        else:
                            # If exact match not found, take the first available QPS data
                            if qps_data:
                                first_thread_count = list(qps_data.keys())[0]
                                qps_info = qps_data[first_thread_count]
                                results[dataset][B][num_threads].update(qps_info)
                                success_messages.append(f"QPS: {qps_info['qps']:.2f} (from T={first_thread_count})")
                                all_logs.append(f"SUCCESS: QPS for B={B}, T={num_threads}: {qps_info['qps']:.2f} (from T={first_thread_count})\n")

                    # Print concise success message
                    if success_messages:
                        print(f"  ✅ {' | '.join(success_messages)}")
                else:
                    print(f"  ❌ Could not parse data for B={B}, num_threads={num_threads}")
                    all_logs.append(f"WARNING: Could not parse data for B={B}, num_threads={num_threads}\n")
                    # Save the raw report for manual inspection
                    with open(vtune_results_dir / f"report_{result_name}.txt", "w") as f:
                        f.write(report_stdout)
                    print(f"  📄 Raw report saved to: report_{result_name}.txt")
                    all_logs.append(f"Raw report saved to: {vtune_results_dir / f'report_{result_name}.txt'}\n")

                print()  # Add space between measurements

            # Save intermediate results after completing each B value
            print(f"🔄 Completed B = {B}, saving intermediate results...")
            save_results_and_logs(results, all_logs, results_file, log_file,
                                 datasets, B_values, num_threads_values, group, base_args, is_partial=True)
            print(f"✅ Intermediate results saved for B = {B}")
            print(f"{'='*60}\n")

    # Print final results
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*60)

    for dataset in results:
        print(f"\n📊 Dataset: {dataset}")
        print("-" * 50)
        if results[dataset]:
            for B in sorted(results[dataset].keys()):
                print(f"\n  🔧 B = {B}:")
                if results[dataset][B]:
                    for num_threads in sorted(results[dataset][B].keys()):
                        data = results[dataset][B][num_threads]
                        line_parts = [f"    T={num_threads:2d}:"]

                        if 'memory_bandwidth_gbps' in data:
                            line_parts.append(f"BW={data['memory_bandwidth_gbps']:6.2f} GB/s")
                        if 'qps' in data:
                            line_parts.append(f"QPS={data['qps']:8.2f}")
                        if 'recall' in data:
                            line_parts.append(f"Recall={data['recall']:.3f}")
                        if 'avg_tm_ms' in data:
                            line_parts.append(f"AvgTime={data['avg_tm_ms']:.2f}ms")

                        print(" | ".join(line_parts))
                else:
                    print("    ❌ No successful measurements")
        else:
            print("❌ No successful measurements")

    print(f"\n💾 Results saved to: {results_file}")
    print(f"📝 Full execution log saved to: {log_file}")
    print("="*60)

    # Save final results to JSON file and logs
    save_results_and_logs(results, all_logs, results_file, log_file,
                         datasets, B_values, num_threads_values, group, base_args, is_partial=False)

if __name__ == "__main__":
    main()
