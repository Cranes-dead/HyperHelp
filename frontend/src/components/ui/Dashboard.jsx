import React from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { 
  ShoppingBag, 
  Home, 
  TrendingUp, 
  DollarSign,
  Package,
  Users,
  FileText,
  XCircle,
  RotateCcw,
  Calendar
} from 'lucide-react';

const salesData = [
  { month: 'Jan', Purchase: 55000, Sales: 48000 },
  { month: 'Feb', Purchase: 58000, Sales: 46000 },
  { month: 'Mar', Purchase: 43000, Sales: 52000 },
  { month: 'Apr', Purchase: 35000, Sales: 42000 },
  { month: 'May', Purchase: 42000, Sales: 45000 },
  { month: 'Jun', Purchase: 27000, Sales: 40000 },
  { month: 'Jul', Purchase: 55000, Sales: 48000 },
  { month: 'Aug', Purchase: 45000, Sales: 42000 },
  { month: 'May', Purchase: 43000, Sales: 43000 },
  { month: 'Jun', Purchase: 36000, Sales: 42000 },
];

const orderData = [
  { month: 'Jan', Ordered: 3800, Delivered: 2000 },
  { month: 'Feb', Ordered: 2800, Delivered: 3600 },
  { month: 'Mar', Ordered: 2200, Delivered: 3500 },
  { month: 'Apr', Ordered: 2000, Delivered: 2800 },
  { month: 'May', Ordered: 2200, Delivered: 3400 },
];

const Dashboard = () => {
  return (
    <div className="dashboard">
      {/* First Row */}
      <div className="dashboard-row">
        <div className="card sales-overview">
          <h2>Sales Overview</h2>
          <div className="stats-container">
            <div className="stat-item">
              <div className="stat-icon blue">
                <Package size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">₹ 832</span>
                <span className="stat-label">Sales</span>
              </div>
            </div>
            <div className="stat-item">
              <div className="stat-icon purple">
                <TrendingUp size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">₹ 18,300</span>
                <span className="stat-label">Revenue</span>
              </div>
            </div>
            <div className="stat-item">
              <div className="stat-icon orange">
                <DollarSign size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">₹ 868</span>
                <span className="stat-label">Profit</span>
              </div>
            </div>
            <div className="stat-item">
              <div className="stat-icon green">
                <Home size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">₹ 17,432</span>
                <span className="stat-label">Cost</span>
              </div>
            </div>
          </div>
        </div>

        <div className="card inventory-summary">
          <h2>Inventory Summary</h2>
          <div className="inventory-stats">
            <div className="inventory-stat">
              <div className="stat-icon orange">
                <Package size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">868</span>
                <span className="stat-label">Quantity in Hand</span>
              </div>
            </div>
            <div className="inventory-stat">
              <div className="stat-icon purple">
                <ShoppingBag size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">200</span>
                <span className="stat-label">To be received</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Second Row */}
      <div className="dashboard-row">
        <div className="card purchase-overview">
          <h2>Purchase Overview</h2>
          <div className="stats-container">
            <div className="stat-item">
              <div className="stat-icon blue">
                <ShoppingBag size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">82</span>
                <span className="stat-label">Purchase</span>
              </div>
            </div>
            <div className="stat-item">
              <div className="stat-icon green">
                <Home size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">₹ 13,573</span>
                <span className="stat-label">Cost</span>
              </div>
            </div>
            <div className="stat-item">
              <div className="stat-icon purple">
                <XCircle size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">5</span>
                <span className="stat-label">Cancel</span>
              </div>
            </div>
            <div className="stat-item">
              <div className="stat-icon orange">
                <RotateCcw size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">₹17,432</span>
                <span className="stat-label">Return</span>
              </div>
            </div>
          </div>
        </div>

        <div className="card product-summary">
          <h2>Product Summary</h2>
          <div className="product-stats">
            <div className="product-stat">
              <div className="stat-icon blue">
                <Users size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">31</span>
                <span className="stat-label">Number of Suppliers</span>
              </div>
            </div>
            <div className="product-stat">
              <div className="stat-icon purple">
                <FileText size={24} />
              </div>
              <div className="stat-info">
                <span className="stat-value">21</span>
                <span className="stat-label">Number of Categories</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Third Row */}
      <div className="dashboard-row">
        <div className="card sales-purchase">
          <div className="card-header">
            <h2>Sales & Purchase</h2>
            <div className="period-selector">
              <Calendar size={16} />
              <span>Weekly</span>
            </div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={salesData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Purchase" fill="#60A5FA" />
                <Bar dataKey="Sales" fill="#4ADE80" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card order-summary">
          <h2>Order Summary</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={orderData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="Ordered" stroke="#F59E0B" />
                <Line type="monotone" dataKey="Delivered" stroke="#60A5FA" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;