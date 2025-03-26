import { getBusinessObject } from 'bpmn-js/lib/util/ModelUtil';
import { isUndefined } from 'min-dash';
import localTemplates from './templates.json'; // Adjust path as needed

export default class LocalAPIProvider {
  constructor(config = {}, eventBus, refactorings, elementTemplates) {
    refactorings.registerProvider(this);
    this._eventBus = eventBus;
    this._config = config;
    this._elementTemplates = elementTemplates;

    // Initialize templates (optional, depending on your needs)
    this._initializeTemplates();
  }

  async _initializeTemplates() {
    // Optional: Add logic here if you want to send templates to a backend
    console.log('Initializing templates...');
  }

  async getRefactorings(elements) {
    if (elements.length !== 1) return [];
    const element = elements[0];
    const businessObject = getBusinessObject(element);
    const name = businessObject.get('name');
    const type = businessObject.$type;

    if (isUndefined(name) || !name.trim()) return [];

    try {
      const suggestions = await this._fetchSuggestions(name, type);
      return suggestions.map(suggestion => ({
        id: `template-${suggestion.id}`,
        label: `Apply ${suggestion.name} (Score: ${suggestion.similarity.toFixed(2)})`,
        execute: (elements) => {
          if (elements.length !== 1) throw new Error(`Expected one element, got ${elements.length}`);
          this._applyTemplate(elements[0], suggestion.id);
        }
      }));
    } catch (error) {
      console.error('Error fetching suggestions:', error);
      return [];
    }
  }

  async _fetchSuggestions(name, type) {
    const response = await fetch('http://localhost:8000/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, type })
    });
    if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
    return response.json();
  }

  _applyTemplate(element, templateId) {
    let template = this._elementTemplates.get(templateId);

    // If not found in elementTemplates, check localTemplates
    if (!template) {
      template = localTemplates.find(t => t.id === templateId);
    }

    if (template) {
      this._eventBus.fire('refactorings.execute', {
        refactoring: { type: 'element-template', elementTemplateId: templateId }
      });
      this._elementTemplates.applyTemplate(element, template);
    } else {
      console.error(`Template with ID ${templateId} not found in either elementTemplates or local templates`);
    }
  }
}

LocalAPIProvider.$inject = ['config.refactorings', 'eventBus', 'refactorings', 'elementTemplates'];