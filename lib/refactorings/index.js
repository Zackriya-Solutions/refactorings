import Refactorings from './Refactorings';
import LocalAPIProvider from './providers/local-api/LocalAPIProvider';

export default {
  __init__: ['refactorings', 'localAPIProvider'],
  refactorings: ['type', Refactorings],
  localAPIProvider: ['type', LocalAPIProvider]
};