#!/usr/bin/env python3
"""
Resource Monitoring Script for MathBoardAI Agent Performance Testing

This script monitors CPU and memory usage of the MathBoardAI Agent application
during performance testing, logging the data to CSV for analysis.

Author: MathBoardAI Agent Team
Task ID: TEST-003
"""

import psutil
import time
import csv
import os
import sys
import logging
import argparse
import signal
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Monitors system resources (CPU, Memory, Network, Disk) for a specific process
    or Docker container and logs the data to CSV files.
    """
    
    def __init__(self, 
                 output_file: str = "resource_log.csv",
                 interval: float = 1.0,
                 container_name: Optional[str] = None,
                 process_name: Optional[str] = None,
                 process_id: Optional[int] = None):
        """
        Initialize the resource monitor.
        
        Args:
            output_file: Path to CSV output file
            interval: Monitoring interval in seconds
            container_name: Docker container name to monitor
            process_name: Process name to monitor
            process_id: Specific process ID to monitor
        """
        self.output_file = output_file
        self.interval = interval
        self.container_name = container_name
        self.process_name = process_name
        self.process_id = process_id
        
        self.monitoring = False
        self.monitor_thread = None
        self.process = None
        self.docker_client = None
        self.container = None
        
        # Data storage
        self.data_points = []
        self.start_time = None
        
        # CSV headers
        self.csv_headers = [
            'timestamp',
            'elapsed_seconds',
            'cpu_percent',
            'memory_mb',
            'memory_percent',
            'memory_available_mb',
            'disk_read_mb',
            'disk_write_mb',
            'network_sent_mb',
            'network_recv_mb',
            'num_threads',
            'num_fds',
            'status'
        ]
        
        logger.info(f"Resource monitor initialized - Output: {self.output_file}, Interval: {self.interval}s")
    
    def _find_process(self) -> Optional[psutil.Process]:
        """
        Find the target process to monitor.
        
        Returns:
            psutil.Process object or None if not found
        """
        try:
            # If process ID is specified, use it directly
            if self.process_id:
                try:
                    process = psutil.Process(self.process_id)
                    logger.info(f"Found process by PID: {self.process_id} - {process.name()}")
                    return process
                except psutil.NoSuchProcess:
                    logger.error(f"Process with PID {self.process_id} not found")
                    return None
            
            # If process name is specified, search for it
            if self.process_name:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if self.process_name.lower() in proc.info['name'].lower():
                            logger.info(f"Found process by name: {proc.info['pid']} - {proc.info['name']}")
                            return psutil.Process(proc.info['pid'])
                        
                        # Also check command line for python scripts
                        if proc.info['cmdline']:
                            cmdline = ' '.join(proc.info['cmdline']).lower()
                            if self.process_name.lower() in cmdline:
                                logger.info(f"Found process by cmdline: {proc.info['pid']} - {cmdline}")
                                return psutil.Process(proc.info['pid'])
                                
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                logger.warning(f"Process '{self.process_name}' not found")
                return None
            
            # If no specific process, monitor system-wide
            logger.info("No specific process specified, monitoring system-wide resources")
            return None
            
        except Exception as e:
            logger.error(f"Error finding process: {e}")
            return None
    
    def _find_container(self) -> Optional[docker.models.containers.Container]:
        """
        Find the target Docker container to monitor.
        
        Returns:
            Docker container object or None if not found
        """
        if not self.container_name:
            return None
            
        try:
            self.docker_client = docker.from_env()
            
            # Try to find running container
            try:
                container = self.docker_client.containers.get(self.container_name)
                if container.status == 'running':
                    logger.info(f"Found running container: {self.container_name} (ID: {container.short_id})")
                    return container
                else:
                    logger.warning(f"Container {self.container_name} found but not running (status: {container.status})")
                    return None
            except docker.errors.NotFound:
                logger.warning(f"Container '{self.container_name}' not found")
                return None
                
        except Exception as e:
            logger.error(f"Error connecting to Docker: {e}")
            return None
    
    def _get_container_stats(self) -> Dict:
        """
        Get resource statistics from Docker container.
        
        Returns:
            Dictionary containing resource statistics
        """
        if not self.container:
            return {}
        
        try:
            # Get container stats (non-blocking)
            stats = self.container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_percent = 0.0
            if 'cpu_stats' in stats and 'precpu_stats' in stats:
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * \
                                 len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
            
            # Get memory statistics
            memory_usage = stats.get('memory_stats', {}).get('usage', 0)
            memory_limit = stats.get('memory_stats', {}).get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            # Get network statistics
            networks = stats.get('networks', {})
            network_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
            network_tx = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            # Get block I/O statistics
            blkio_stats = stats.get('blkio_stats', {})
            io_read = sum(
                item.get('value', 0) for item in blkio_stats.get('io_service_bytes_recursive', [])
                if item.get('op') == 'Read'
            )
            io_write = sum(
                item.get('value', 0) for item in blkio_stats.get('io_service_bytes_recursive', [])
                if item.get('op') == 'Write'
            )
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_mb': round(memory_usage / (1024 * 1024), 2),
                'memory_percent': round(memory_percent, 2),
                'memory_available_mb': round((memory_limit - memory_usage) / (1024 * 1024), 2),
                'network_sent_mb': round(network_tx / (1024 * 1024), 2),
                'network_recv_mb': round(network_rx / (1024 * 1024), 2),
                'disk_read_mb': round(io_read / (1024 * 1024), 2),
                'disk_write_mb': round(io_write / (1024 * 1024), 2),
                'num_threads': 0,  # Not available for containers
                'num_fds': 0,      # Not available for containers
                'status': 'running'
            }
            
        except Exception as e:
            logger.error(f"Error getting container stats: {e}")
            return {}
    
    def _get_process_stats(self) -> Dict:
        """
        Get resource statistics from process.
        
        Returns:
            Dictionary containing resource statistics
        """
        if not self.process:
            # Return system-wide statistics
            try:
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                return {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_mb': round((memory.total - memory.available) / (1024 * 1024), 2),
                    'memory_percent': memory.percent,
                    'memory_available_mb': round(memory.available / (1024 * 1024), 2),
                    'disk_read_mb': round(disk_io.read_bytes / (1024 * 1024), 2) if disk_io else 0,
                    'disk_write_mb': round(disk_io.write_bytes / (1024 * 1024), 2) if disk_io else 0,
                    'network_sent_mb': round(network_io.bytes_sent / (1024 * 1024), 2) if network_io else 0,
                    'network_recv_mb': round(network_io.bytes_recv / (1024 * 1024), 2) if network_io else 0,
                    'num_threads': 0,
                    'num_fds': 0,
                    'status': 'system'
                }
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return {}
        
        try:
            # Check if process still exists
            if not self.process.is_running():
                logger.warning("Target process is no longer running")
                return {}
            
            # Get process statistics
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            # Get system memory for percentage calculation
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_info.rss / system_memory.total) * 100
            
            # Get I/O statistics (if available)
            try:
                io_counters = self.process.io_counters()
                disk_read_mb = round(io_counters.read_bytes / (1024 * 1024), 2)
                disk_write_mb = round(io_counters.write_bytes / (1024 * 1024), 2)
            except (psutil.AccessDenied, AttributeError):
                disk_read_mb = 0
                disk_write_mb = 0
            
            # Get number of threads and file descriptors
            try:
                num_threads = self.process.num_threads()
            except (psutil.AccessDenied, AttributeError):
                num_threads = 0
            
            try:
                num_fds = self.process.num_fds()
            except (psutil.AccessDenied, AttributeError):
                num_fds = 0
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_mb': round(memory_info.rss / (1024 * 1024), 2),
                'memory_percent': round(memory_percent, 2),
                'memory_available_mb': round(system_memory.available / (1024 * 1024), 2),
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'network_sent_mb': 0,  # Process-level network stats not easily available
                'network_recv_mb': 0,
                'num_threads': num_threads,
                'num_fds': num_fds,
                'status': self.process.status()
            }
            
        except psutil.NoSuchProcess:
            logger.warning("Target process no longer exists")
            return {}
        except Exception as e:
            logger.error(f"Error getting process stats: {e}")
            return {}
    
    def _collect_data_point(self) -> Optional[Dict]:
        """
        Collect a single data point with current resource usage.
        
        Returns:
            Dictionary containing timestamp and resource data
        """
        current_time = datetime.now()
        elapsed_seconds = (current_time - self.start_time).total_seconds() if self.start_time else 0
        
        # Get stats from appropriate source
        if self.container:
            stats = self._get_container_stats()
        else:
            stats = self._get_process_stats()
        
        if not stats:
            return None
        
        # Create data point
        data_point = {
            'timestamp': current_time.isoformat(),
            'elapsed_seconds': round(elapsed_seconds, 2),
            **stats
        }
        
        return data_point
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        logger.info("Starting resource monitoring loop")
        
        # Initialize CSV file
        self._initialize_csv()
        
        while self.monitoring:
            try:
                data_point = self._collect_data_point()
                
                if data_point:
                    self.data_points.append(data_point)
                    self._write_data_point(data_point)
                    
                    # Log periodic updates
                    if len(self.data_points) % 30 == 0:  # Every 30 intervals
                        logger.info(f"Monitoring... CPU: {data_point['cpu_percent']}%, "
                                   f"Memory: {data_point['memory_mb']}MB "
                                   f"({data_point['memory_percent']:.1f}%)")
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
        
        logger.info("Resource monitoring loop stopped")
    
    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                writer.writeheader()
            logger.info(f"CSV file initialized: {self.output_file}")
        except Exception as e:
            logger.error(f"Error initializing CSV file: {e}")
    
    def _write_data_point(self, data_point: Dict):
        """Write a single data point to the CSV file."""
        try:
            with open(self.output_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                writer.writerow(data_point)
        except Exception as e:
            logger.error(f"Error writing to CSV file: {e}")
    
    def start_monitoring(self) -> bool:
        """
        Start the resource monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self.monitoring:
            logger.warning("Monitoring is already running")
            return False
        
        # Find target process or container
        if self.container_name:
            self.container = self._find_container()
            if not self.container:
                logger.error("Failed to find target container")
                return False
        else:
            self.process = self._find_process()
            # Note: self.process can be None for system-wide monitoring
        
        self.monitoring = True
        self.start_time = datetime.now()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
        return True
    
    def stop_monitoring(self):
        """Stop the resource monitoring."""
        if not self.monitoring:
            logger.warning("Monitoring is not running")
            return
        
        logger.info("Stopping resource monitoring...")
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Generate summary
        self._generate_summary()
        
        logger.info(f"Resource monitoring stopped. Data saved to {self.output_file}")
    
    def _generate_summary(self):
        """Generate and log a summary of the monitoring session."""
        if not self.data_points:
            logger.info("No data points collected")
            return
        
        # Calculate statistics
        cpu_values = [dp['cpu_percent'] for dp in self.data_points if dp['cpu_percent'] is not None]
        memory_values = [dp['memory_mb'] for dp in self.data_points if dp['memory_mb'] is not None]
        
        if cpu_values:
            cpu_avg = sum(cpu_values) / len(cpu_values)
            cpu_max = max(cpu_values)
            cpu_min = min(cpu_values)
        else:
            cpu_avg = cpu_max = cpu_min = 0
        
        if memory_values:
            memory_avg = sum(memory_values) / len(memory_values)
            memory_max = max(memory_values)
            memory_min = min(memory_values)
        else:
            memory_avg = memory_max = memory_min = 0
        
        duration = self.data_points[-1]['elapsed_seconds'] if self.data_points else 0
        
        logger.info("=== MONITORING SUMMARY ===")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Data points collected: {len(self.data_points)}")
        logger.info(f"CPU Usage - Avg: {cpu_avg:.1f}%, Max: {cpu_max:.1f}%, Min: {cpu_min:.1f}%")
        logger.info(f"Memory Usage - Avg: {memory_avg:.1f}MB, Max: {memory_max:.1f}MB, Min: {memory_min:.1f}MB")
        
        # Check for potential issues
        if memory_max > 1000:  # 1GB
            logger.warning(f"High memory usage detected: {memory_max:.1f}MB")
        
        if cpu_avg > 80:
            logger.warning(f"High average CPU usage: {cpu_avg:.1f}%")


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {signum}, stopping monitoring...")
    global monitor
    if monitor:
        monitor.stop_monitoring()
    sys.exit(0)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Monitor system resources for MathBoardAI Agent")
    
    parser.add_argument(
        "--output", "-o",
        default="resource_log.csv",
        help="Output CSV file (default: resource_log.csv)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Monitoring interval in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--container", "-c",
        help="Docker container name to monitor"
    )
    
    parser.add_argument(
        "--process", "-p",
        help="Process name to monitor"
    )
    
    parser.add_argument(
        "--pid",
        type=int,
        help="Process ID to monitor"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Monitoring duration in seconds (default: monitor until interrupted)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start monitor
    global monitor
    monitor = ResourceMonitor(
        output_file=args.output,
        interval=args.interval,
        container_name=args.container,
        process_name=args.process,
        process_id=args.pid
    )
    
    if not monitor.start_monitoring():
        logger.error("Failed to start monitoring")
        sys.exit(1)
    
    try:
        if args.duration:
            logger.info(f"Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
            monitor.stop_monitoring()
        else:
            logger.info("Monitoring until interrupted (Ctrl+C to stop)...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()