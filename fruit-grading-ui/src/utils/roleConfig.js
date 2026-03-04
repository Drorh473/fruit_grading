// Role-based access and navigation configuration

export const ROLES = {
  ADMIN: 'admin',
  USER: 'user'
};

export const ROLE_PERMISSIONS = {
  admin: {
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
    canAccess: [
      '/user-dashboard',
      '/results'
    ],
    features: {
      viewSystemStatus: false,
      viewCameras: false,
      runProcessing: false,
      viewResults: true,
      addFruit: false,
      modifySettings: false,
      exportData: true,
      viewLogs: false,
      manageDatabase: false
    }
  }
};

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

export const hasAccess = (userRole, path) => {
  if (!userRole || !ROLE_PERMISSIONS[userRole]) {
    return false;
  }
  return ROLE_PERMISSIONS[userRole].canAccess.includes(path);
};

export const hasFeaturePermission = (userRole, feature) => {
  if (!userRole || !ROLE_PERMISSIONS[userRole]) {
    return false;
  }
  return ROLE_PERMISSIONS[userRole].features[feature] || false;
};

export default ROLE_PERMISSIONS;
