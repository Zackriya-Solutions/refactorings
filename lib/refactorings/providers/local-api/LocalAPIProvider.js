import { getBusinessObject } from 'bpmn-js/lib/util/ModelUtil';
import { isUndefined } from 'min-dash';

export default class LocalAPIProvider {
  constructor(config = {}, eventBus, refactorings) {
    refactorings.registerProvider(this);
    this._eventBus = eventBus;
    this._config = config;
  }

  async getRefactorings(elements) {
    // Support single-element selection
    if (elements.length !== 1) {
      return [];
    }

    const element = elements[0];
    const businessObject = getBusinessObject(element);
    const name = businessObject.get('name');
    const type = businessObject.$type;

    // Skip if name is undefined or empty
    if (isUndefined(name) || !name.trim()) {
      return [];
    }

    try {
      const suggestions = await this._fetchSuggestions(name, type);
      return suggestions.map(suggestion => ({
        id: `template-${suggestion.id}`,
        label: `Apply ${suggestion.name} (Score: ${suggestion.similarity.toFixed(2)})`,
        execute: (elements) => {
          if (elements.length !== 1) {
            throw new Error(`Expected one element, got ${elements.length}`);
          }
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

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return response.json();
  }

  _applyTemplate(element, templateId) {
    const elementTemplates = this._config.elementTemplates;
    const template = elementTemplates.get(templateId);
    if (template) {
      elementTemplates.applyTemplate(element, template);
    } else {
      console.error(`Template with ID ${templateId} not found`);
    }
  }
}

LocalAPIProvider.$inject = ['config.refactorings', 'eventBus', 'refactorings'];