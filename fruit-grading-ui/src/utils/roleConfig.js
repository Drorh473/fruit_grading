// Role-based access configuration

export const ROLES = {
  ADMIN: 'admin',
  USER: 'user'
};

// Define which pages each role can access
export const ROLE_PERMISSIONS = {
  admin: {
    // Admin has full access to everything
    canAccess: [
      '/dashboard',
      '/cameras',
      '/processing',
      '/results',
      '/add-fruit',
      '/settings'
    ],
    features: {
      viewSystemStatus: true,
      viewCameras: true,
      runProcessing: true,
      viewResults: true,
      addFruit: true,
      modifySettings: true,
      exportData: true,
      viewLogs: true,
      manageDatabase: true
    }
  },
  user: {
    // User has limited access - viewing only
    canAccess: [
      '/user-dashboard',
      '/results'
    ],
    features: {
      viewSystemStatus: false,     // Can't see technical status
      viewCameras: false,           // Can't access camera monitor
      runProcessing: false,         // Can't run pipeline
      viewResults: true,            // Can view results only
      addFruit: false,              // Can't add new fruits
      modifySettings: false,        // Can't change settings
      exportData: true,             // Can export their results
      viewLogs: false,              // Can't see system logs
      manageDatabase: false         // Can't manage database
    }
  }
};

// Navigation items for each role
export const NAVIGATION_BY_ROLE = {
  admin: [
    { path: '/dashboard', label: 'Dashboard', icon: 'FiHome' },
    { path: '/cameras', label: 'Camera Monitor', icon: 'FiCamera' },
    { path: '/processing', label: 'Processing', icon: 'FiCpu' },
    { path: '/results', label: 'Results', icon: 'FiBarChart2' },
    { path: '/add-fruit', label: 'Add Fruit', icon: 'FiPlusCircle' },
    { path: '/settings', label: 'Settings', icon: 'FiSettings' }
  ],
  user: [
    { path: '/user-dashboard', label: 'Dashboard', icon: 'FiHome' },
    { path: '/results', label: 'View Results', icon: 'FiBarChart2' }
  ]
};

// Helper function to check if user has access to a route
export const hasAccess = (userRole, path) => {
  if (!userRole || !ROLE_PERMISSIONS[userRole]) {
    return false;
  }
  return ROLE_PERMISSIONS[userRole].canAccess.includes(path);
};

// Helper function to check if user has a specific feature permission
export const hasFeaturePermission = (userRole, feature) => {
  if (!userRole || !ROLE_PERMISSIONS[userRole]) {
    return false;
  }
  return ROLE_PERMISSIONS[userRole].features[feature] || false;
};

export default ROLE_PERMISSIONS;
