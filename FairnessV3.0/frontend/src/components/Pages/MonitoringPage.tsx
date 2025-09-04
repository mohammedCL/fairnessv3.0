import React, { useState, useEffect } from 'react';
import { Eye, TrendingUp, AlertTriangle, CheckCircle, RefreshCw, Download, Calendar, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { useFairness } from '../../context/FairnessContext';
import clsx from 'clsx';

interface MonitoringMetric {
  timestamp: string;
  metric_name: string;
  value: number;
  threshold: number;
  status: 'good' | 'warning' | 'alert';
}

interface Alert {
  id: string;
  timestamp: string;
  type: 'bias_increase' | 'fairness_degradation' | 'data_drift';
  severity: 'low' | 'medium' | 'high';
  message: string;
  metric_name: string;
  value: number;
  threshold: number;
}

const MonitoringPage: React.FC = () => {
  const { state } = useFairness();
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [selectedMetric, setSelectedMetric] = useState<string>('all');
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Mock monitoring data - in real app this would come from API
  const [monitoringData, setMonitoringData] = useState<MonitoringMetric[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    // Generate mock monitoring data
    generateMockData();
  }, [timeRange]);

  const generateMockData = () => {
    const metrics = ['demographic_parity', 'equal_opportunity', 'calibration', 'statistical_parity'];
    const data: MonitoringMetric[] = [];
    const alertsData: Alert[] = [];
    
    const now = new Date();
    const intervals = timeRange === '1h' ? 12 : timeRange === '24h' ? 24 : timeRange === '7d' ? 7 : 30;
    const intervalMs = timeRange === '1h' ? 5 * 60 * 1000 : 
                     timeRange === '24h' ? 60 * 60 * 1000 : 
                     timeRange === '7d' ? 24 * 60 * 60 * 1000 : 
                     24 * 60 * 60 * 1000;

    for (let i = intervals; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * intervalMs).toISOString();
      
      metrics.forEach(metric => {
        const baseValue = Math.random() * 0.3 + 0.1;
        const drift = (intervals - i) * 0.002; // Simulate gradual drift
        const noise = (Math.random() - 0.5) * 0.05;
        const value = baseValue + drift + noise;
        const threshold = 0.2;
        
        const status: 'good' | 'warning' | 'alert' = 
          value < threshold * 0.8 ? 'good' :
          value < threshold ? 'warning' : 'alert';

        data.push({
          timestamp,
          metric_name: metric,
          value,
          threshold,
          status
        });

        // Generate alerts for high values
        if (value > threshold && Math.random() > 0.7) {
          alertsData.push({
            id: `${metric}_${timestamp}`,
            timestamp,
            type: 'bias_increase',
            severity: value > threshold * 1.5 ? 'high' : 'medium',
            message: `${metric.replace(/_/g, ' ')} exceeded threshold`,
            metric_name: metric,
            value,
            threshold
          });
        }
      });
    }

    setMonitoringData(data);
    setAlerts(alertsData.slice(0, 10)); // Keep recent alerts
  };

  const refreshData = async () => {
    setIsRefreshing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    generateMockData();
    setIsRefreshing(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      case 'warning': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'alert': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-400';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-400';
    }
  };

  const filteredData = selectedMetric === 'all' 
    ? monitoringData 
    : monitoringData.filter(d => d.metric_name === selectedMetric);

  const chartData = filteredData.reduce((acc, curr) => {
    const timeKey = new Date(curr.timestamp).toLocaleTimeString();
    const existing = acc.find(item => item.time === timeKey);
    
    if (existing) {
      existing[curr.metric_name] = curr.value;
    } else {
      acc.push({
        time: timeKey,
        [curr.metric_name]: curr.value
      });
    }
    
    return acc;
  }, [] as any[]);

  const currentStatus = {
    overall: monitoringData.length > 0 ? 
      monitoringData.slice(-4).every(d => d.status === 'good') ? 'good' :
      monitoringData.slice(-4).some(d => d.status === 'alert') ? 'alert' : 'warning' : 'good',
    alerts: alerts.filter(a => new Date(a.timestamp) > new Date(Date.now() - 24 * 60 * 60 * 1000)).length,
    uptime: 99.7,
    lastUpdate: new Date().toLocaleString()
  };

  if (!state.mitigationResults) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center py-12">
          <Eye className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Monitoring Not Available
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Complete the mitigation process to access ongoing fairness monitoring.
          </p>
          <button
            onClick={() => window.location.href = '/mitigation'}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
          >
            Go to Mitigation
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Fairness Monitoring
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Continuous monitoring of model fairness and bias metrics in production.
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={refreshData}
            disabled={isRefreshing}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <RefreshCw className={clsx('h-4 w-4 mr-2', isRefreshing && 'animate-spin')} />
            Refresh
          </button>
          <button className="inline-flex items-center px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-blue-600 hover:bg-blue-700">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </button>
        </div>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center">
            <div className={clsx(
              'p-3 rounded-lg',
              getStatusColor(currentStatus.overall)
            )}>
              <Activity className="h-6 w-6" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Overall Status</p>
              <p className={clsx(
                'text-2xl font-bold capitalize',
                currentStatus.overall === 'good' ? 'text-green-600' :
                currentStatus.overall === 'warning' ? 'text-yellow-600' :
                'text-red-600'
              )}>
                {currentStatus.overall}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-red-100 dark:bg-red-900/20">
              <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Alerts</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {currentStatus.alerts}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/20">
              <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Uptime</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {currentStatus.uptime}%
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/20">
              <Calendar className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Last Update</p>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {currentStatus.lastUpdate}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Fairness Metrics Over Time
          </h3>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Time Range:</span>
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as any)}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Metric:</span>
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
              >
                <option value="all">All Metrics</option>
                <option value="demographic_parity">Demographic Parity</option>
                <option value="equal_opportunity">Equal Opportunity</option>
                <option value="calibration">Calibration</option>
                <option value="statistical_parity">Statistical Parity</option>
              </select>
            </div>
          </div>
        </div>

        <div className="mt-6 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="time" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--tooltip-bg)', 
                  border: '1px solid var(--tooltip-border)',
                  borderRadius: '6px'
                }}
              />
              {selectedMetric === 'all' ? (
                <>
                  <Line type="monotone" dataKey="demographic_parity" stroke="#3B82F6" strokeWidth={2} />
                  <Line type="monotone" dataKey="equal_opportunity" stroke="#10B981" strokeWidth={2} />
                  <Line type="monotone" dataKey="calibration" stroke="#F59E0B" strokeWidth={2} />
                  <Line type="monotone" dataKey="statistical_parity" stroke="#EF4444" strokeWidth={2} />
                </>
              ) : (
                <Line type="monotone" dataKey={selectedMetric} stroke="#3B82F6" strokeWidth={2} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Alerts */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Recent Alerts
        </h3>
        
        {alerts.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No Active Alerts
            </h4>
            <p className="text-gray-600 dark:text-gray-400">
              Your model is performing within expected fairness parameters.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {alerts.map((alert) => (
              <div key={alert.id} className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className={clsx(
                      'h-5 w-5',
                      alert.severity === 'high' ? 'text-red-500' :
                      alert.severity === 'medium' ? 'text-yellow-500' :
                      'text-orange-500'
                    )} />
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      {alert.message}
                    </h4>
                    <span className={clsx(
                      'px-2 py-1 rounded-full text-xs font-medium',
                      getSeverityColor(alert.severity)
                    )}>
                      {alert.severity}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {new Date(alert.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                  <span className="mr-4">
                    <strong>Metric:</strong> {alert.metric_name.replace(/_/g, ' ')}
                  </span>
                  <span className="mr-4">
                    <strong>Value:</strong> {alert.value.toFixed(3)}
                  </span>
                  <span>
                    <strong>Threshold:</strong> {alert.threshold.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Monitoring Summary
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">
              Key Insights
            </h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Your model maintains good fairness metrics overall</li>
              <li>• Some minor bias drift detected in demographic parity</li>
              <li>• Equal opportunity metrics remain stable</li>
              <li>• Consider retraining if alerts persist for 48+ hours</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">
              Recommended Actions
            </h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Monitor demographic parity trends closely</li>
              <li>• Review recent data for potential distribution shifts</li>
              <li>• Consider setting up automated alerts for threshold breaches</li>
              <li>• Schedule weekly fairness assessment reviews</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonitoringPage;
